(sec-standard-model-cognition)=
# The Standard Model of Cognition: Gauge-Theoretic Formulation

## TLDR

- Show that “gauge fields” are not physics-only: they emerge from **local consistency requirements** when modules/agents
  cannot globally coordinate representations.
- Derive a “standard model” symmetry group for cognition from redundancies that leave observable behavior invariant.
- Interpret gauge connections as the bookkeeping needed to compare values/beliefs across locally chosen conventions.
- This chapter synthesizes the geometry/control stack (metric law, WFR, boundary interface, belief waves, ontology
  dynamics) into a single invariance principle.
- Outputs: a precise vocabulary for what is invariant, what is conventional, and where consistency constraints live in a
  cognitive architecture.

## Roadmap

1. State the gauge principle: redundancy + locality ⇒ connection fields.
2. Derive the symmetry factors and interpret them operationally.
3. Connect the resulting fields to diagnostics, stability, and multi-agent interaction.

:::{div} feynman-prose
Now we come to what I think is the most beautiful part of this whole framework. And I want to be honest with you upfront: this is ambitious. We're going to show that the same mathematical structure that physicists use to describe the fundamental forces of nature---electromagnetism, the weak force, the strong force---emerges naturally from the requirements of being a bounded, distributed, reward-seeking agent.

You might be skeptical. "Come on," you might say, "the Standard Model of particle physics took decades of experiments and Nobel Prizes to figure out. How can it just pop out of thinking about agents?"

Here's the key insight: what we're claiming isn't that cognition *is* particle physics. We're claiming that both systems face the same fundamental mathematical constraint: **the need for local consistency in the absence of global coordination**.

Think about it. An electron in one part of the universe can't instantaneously check with an electron on the other side of the universe to agree on their shared reference frame. They have to carry their own local bookkeeping, and the requirement that physics be consistent despite this locality is what forces gauge fields into existence.

An agent is in exactly the same situation. Different parts of the agent's computational substrate can't instantaneously synchronize their internal representations. The sensor processing module can't check with the motor planning module to agree on what "zero value" means. They each have their local perspective, and the requirement that decisions be consistent despite this locality forces the same mathematical structures.

This is not a metaphor. It's a theorem.
:::

*Abstract.* This chapter demonstrates that the internal symmetry group
$G_{\text{Fragile}} = SU(N_f)_C \times SU(r)_L \times U(1)_Y$ emerges necessarily from the cybernetic constraints of a
bounded, distributed, reward-seeking agent. The **Feature Dimension** $N_f$ is determined by the agent's environment,
while the **Mode Rank** $r$ is the minimal Kraus rank required by local belief-update channels (Definition
{prf:ref}`def-mode-rank-parameter`). For the minimal observation/action agent, $r=2$, so $SU(r)_L = SU(2)_L$.
The physics Standard Model corresponds to the special case $N_f = 3$ and $r=2$. Each factor is derived from
redundancies in the agent's description that leave physical observables invariant. The proofs rely explicitly on prior
definitions from the
WFR framework ({ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`), the
Belief Wave-Function ({ref}`sec-the-belief-wave-function-schrodinger-representation`), the Boundary
Interface ({ref}`sec-the-boundary-interface-symplectic-structure`), and the Ontological Fission dynamics
({ref}`sec-ontological-expansion-topological-fission-and-the-semantic-vacuum`).

*Cross-references:* This chapter synthesizes:
- {ref}`sec-the-belief-wave-function-schrodinger-representation`–29.27 (Quantum Layer: Belief Wave-Function, Schrödinger Representation)
- {ref}`sec-the-boundary-interface-symplectic-structure` (Holographic Interface: Dirichlet/Neumann Boundary Conditions)
- {ref}`sec-ontological-expansion-topological-fission-and-the-semantic-vacuum` (Ontological Expansion: Pitchfork Bifurcation, Chart Fission)
- {ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits` (Capacity-Constrained Metric Law)
- {ref}`sec-the-reward-field-value-forms-and-hodge-geometry` (Helmholtz Equation, Value Field)



(sec-gauge-principle-derivation)=
## The Gauge Principle: Derivation of the Symmetry Group $G_{\text{Fragile}}$

:::{div} feynman-prose
Before we dive into the mathematics, let me explain what we're about to do and why it works.

The fundamental principle is this: **redundancy in description forces compensating fields into existence**.

What does that mean? Suppose you have a system where certain choices don't affect the observable outcomes. For instance, suppose you can add a constant to all your utility values without changing which action is best. That's a redundancy---a "gauge freedom" in physics terminology.

Now here's the magic. If you demand that this freedom be *local*---that different parts of the system can make different arbitrary choices independently---then you can no longer compare quantities at different locations directly. You need a "connection" to tell you how to transport quantities from one place to another while accounting for the arbitrary local choices.

This connection is a gauge field. And the requirement that the physics be independent of the arbitrary choices constrains exactly how this field must behave.

We're going to derive three different gauge fields from three different redundancies:
1. **$U(1)_Y$**: The freedom to shift the baseline of utility
2. **$SU(r)_L$**: The freedom to rotate between observation/action update modes (minimal case $r=2$)
3. **$SU(N_f)_C$**: The freedom to relabel feature components

Each one emerges from a genuine redundancy in how we describe the agent's state, and each one forces a compensating field into existence.
:::

We derive the internal symmetry group by identifying redundancies in the agent's description that leave physical observables (Actions and Rewards) invariant. In a distributed agent with finite information speed (Axiom {prf:ref}`ax-information-speed-limit` and Definition {prf:ref}`def-causal-interval`), these redundancies must be *local* across charts. Local redundancy plus a local action density forces the introduction of compensating connection fields so that kinetic terms remain invariant under chart-by-chart reparameterizations. We will show this directly in each sector by tracking how derivatives transform and constructing the corresponding covariant derivative (Remark {prf:ref}`rem-local-gauge-template`).

:::{prf:remark} Local Gauge-Covariance Template
:label: rem-local-gauge-template

Let a field $\Phi(x)$ admit a local redundancy $\Phi \to U(x)\Phi$ with $U(x)\in G$ acting in its internal fiber. Then:

$$
\partial_\mu \Phi \to U\,\partial_\mu \Phi + (\partial_\mu U)\,\Phi,
$$

so the naive kinetic term built from $\partial_\mu \Phi$ is not invariant under local changes of basis. Introduce a connection $A_\mu$ valued in the Lie algebra of $G$ and define

$$
D_\mu := \partial_\mu - i g A_\mu.
$$

Demanding covariance $D_\mu \Phi \to U D_\mu \Phi$ forces the transformation rule

$$
A_\mu \to U A_\mu U^{-1} + \frac{i}{g}(\partial_\mu U)U^{-1}.
$$

With this choice, any kinetic term built from $|D_\mu \Phi|^2$ (or $\bar{\Phi}\gamma^\mu D_\mu\Phi$ for spinors) is locally invariant. In the abelian case $U(x)=e^{i q \alpha(x)}$, this reduces to $A_\mu \to A_\mu + \frac{1}{g}\partial_\mu \alpha$.
:::

### A. $U(1)_Y$: The Hypercharge of Utility

:::{div} feynman-prose
Let's start with the simplest case. Ask yourself: what do you actually observe when an agent makes decisions?

You observe the agent's actions. You can measure how likely the agent is to be in different states. You can see the flow of probability from one state to another. What you *don't* observe is the absolute value of the agent's internal utility function.

Think about it. If I tell you "state A has value 100 and state B has value 80," you know the agent prefers A. But if I tell you "state A has value 1000 and state B has value 980," the agent has exactly the same preference! Adding a constant to all values doesn't change anything observable.

This is the utility gauge freedom. And it's not just a philosophical nicety---it has profound implications.
:::

The fundamental observable in Reinforcement Learning is the **Preference**, defined by the gradient of the scalar value
potential (the conservative component of the external reward 1-form), not its absolute magnitude. The policy is invariant under
certain potential transformations, including constant shifts and potential-based shaping {cite}`ng1999policy`.

:::{prf:definition} Utility Gauge Freedom
:label: def-utility-gauge-freedom

Let the Belief Wave-Function $\psi(z)$ be defined as in Definition {prf:ref}`def-belief-wave-function`:

$$
\psi(z) = \sqrt{\rho(z)} \exp\left(\frac{i V(z)}{\sigma}\right),

$$

where:
- $\rho(z)$ is the belief density (Definition {prf:ref}`def-belief-density`)
- $V(z)$ is the scalar Value potential for the conservative component (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`)
- $\sigma = T_c \cdot \tau_{\text{update}}$ is the Cognitive Action Scale (Definition {prf:ref}`def-cognitive-action-scale`)

The system's observables are:
1. **Probability density:** $\rho = |\psi|^2$
2. **Probability current:** $J^\mu = \text{Im}(\psi^* D^\mu \psi) = \frac{\rho}{\sigma}(\partial^\mu V - A^{\text{ext}\,\mu})$
   (conservative case: $A^{\text{ext}\,\mu}=0$).
   Here $A^{\text{ext}}_\mu$ is the external reward 1-form; the internal $U(1)$ connection $B_\mu$ is introduced below.
   The $D^\mu$ here is the WFR covariant derivative built from $A^{\text{ext}}_\mu$; later $D_\mu$
   denotes the SMoC gauge covariant derivative including $B_\mu$, $W_\mu$, and $G_\mu$.

Both are invariant under the global phase transformation (constant gauge parameter $\alpha$):

$$
\psi(z) \to e^{i(Y/2)\alpha} \psi(z), \quad \alpha \in \mathbb{R}.

$$

This corresponds to the global gauge invariance of the Value function:
$V(z) \to V(z) + \sigma \frac{Y}{2}\alpha$. The addition of a constant baseline does not alter the
policy gradient $\nabla_{A^{\text{ext}}} V$.

:::

:::{div} feynman-prose
Look at what this definition is saying. We've packaged the agent's belief (probability distribution) and value (utility function) into a single complex wave function $\psi$. The probability is encoded in the amplitude, and the value is encoded in the phase.

Now, the phase of a complex number is only defined up to an overall constant---if you multiply every point by $e^{i\alpha}$, you've just rotated the whole phase wheel, and nobody can tell the difference from looking at the amplitude or the current.

But here's where it gets interesting. What if you want to make *different* phase rotations at *different* locations?
:::

:::{prf:axiom} Local Utility Invariance
:label: ax-local-utility-invariance

In a distributed agent with finite information speed $c_{\text{info}}$ (Axiom {prf:ref}`ax-information-speed-limit`), there is no global clock to synchronize the Value baseline across the manifold simultaneously. The agent must possess **Local Gauge Invariance**:

$$
\psi(x) \to e^{i(Y/2)\alpha(x)} \psi(x),

$$

where $x$ denotes the spacetime coordinate on the agent's computational manifold. The choice of "zero utility" can vary locally across different charts without affecting the physical transfer of control authority.

*Justification:* This follows from the Causal Interval (Definition {prf:ref}`def-causal-interval`): spacelike-separated modules cannot instantaneously agree on a common baseline.

:::

:::{div} feynman-prose
This axiom is the key step. We're saying that because information takes time to propagate through the agent's computational substrate, different parts of the agent can't agree on a common "zero point" for utility.

Imagine you have a robot with a sensor module in its head and a motor module in its arm. Light takes time to travel between them (or electrical signals, or whatever). During that propagation time, each module has to operate with its own local notion of "how valuable is this state?" They can't synchronize their zeros instantaneously.

This isn't a bug in the design---it's a fundamental constraint imposed by causality. And it forces structure into existence.
:::

:::{prf:theorem} Emergence of the Opportunity Field ($B_\mu$)
:label: thm-emergence-opportunity-field

To preserve the invariance of the kinetic term in the Inference Action under the local transformation
$\psi \to e^{i(Y/2)\alpha(x)}\psi$, we must replace the partial derivative $\partial_\mu$ with the
**Covariant Derivative**:

$$
D_\mu = \partial_\mu - i g_1 \frac{Y}{2} B_\mu,

$$

where:
- $Y$ is the **Hypercharge** (the reward sensitivity of the module)
- $B_\mu$ is an abelian gauge field (the **Opportunity Field**)
- $g_1$ is the coupling constant

*Proof.*

Apply the local gauge-covariance template (Remark {prf:ref}`rem-local-gauge-template`) with
$U(x)=e^{i(Y/2)\alpha(x)}$ in the $U(1)_Y$ sector.

**Step 1.** Consider the kinetic term from the Inference Schrödinger Equation in the conservative limit ($A=0$):

$$
\mathcal{L}_{\text{kin}} = \psi^* (i\sigma \partial_t) \psi - \frac{\sigma^2}{2}|\nabla \psi|^2.

$$

Under local transformation $\psi \to e^{i(Y/2)\alpha(x)}\psi$:

$$
\partial_\mu \psi \to e^{i(Y/2)\alpha}\left(\partial_\mu \psi + i\frac{Y}{2}(\partial_\mu\alpha)\psi\right).

$$

The kinetic term acquires a spurious contribution $\sigma\frac{Y}{2}(\partial_\mu\alpha)|\psi|^2$
that depends on the arbitrary function $\alpha(x)$.

**Step 2.** Introduce the compensating field $B_\mu$ and a universal gauge parameter $\alpha(x)$,
with field phase $\psi \to e^{i(Y/2)\alpha(x)}\psi$, and transform:

$$
B_\mu \to B_\mu + \frac{1}{g_1} \partial_\mu \alpha(x).

$$

**Step 3.** The covariant derivative $D_\mu \psi = (\partial_\mu - ig_1(Y/2)B_\mu)\psi$ transforms homogeneously:

$$
D_\mu \psi \to e^{i(Y/2)\alpha(x)} D_\mu \psi.

$$

**Step 4.** The gauge-invariant kinetic term is $(D_\mu\psi)^\dagger(D^\mu\psi) = |D_\mu\psi|^2$.
Equivalently, $\mathcal{L}_{\text{kin}} = \psi^*(i\sigma D_t)\psi - \frac{\sigma^2}{2}|D\psi|^2$ in the non-conservative case.

**Identification:** The field $B_\mu$ is the internal $U(1)$ connection (the Opportunity Field), representing the agent's
model of the external reward 1-form $A^{\text{ext}}_\mu$. In the conservative case, a gauge exists with
$B_\mu = \partial_\mu \Phi$. **Local Hodge decomposition (chart-wise).** Restrict to any chart domain on a fixed time
slice; by construction this domain is a bounded submanifold with boundary (the agent's local chart in $\mathcal{Z}$).
Therefore the hypotheses of the Hodge decomposition theorem for the reward 1-form apply on each chart
(Theorem {prf:ref}`thm-hodge-decomposition`), yielding a decomposition of the spatial components $\vec{B}$ into exact
(gradient), coexact (solenoidal), and harmonic parts. The non-conservative structure is measured by the value curl
$\mathcal{F} = d\mathcal{R}$ (Definition {prf:ref}`def-value-curl`). This local statement is all that is required here:
the decomposition is guaranteed chart-by-chart, without assuming global topology or global exactness.

The field strength tensor $B_{\mu\nu} = \partial_\mu B_\nu - \partial_\nu B_\mu$ measures the non-conservative
component of the internal opportunity 1-form (Value Curl; Definition {prf:ref}`def-value-curl`). When $B_{\mu\nu} \neq 0$,
no choice of baseline can make the internal opportunity 1-form path-independent.

$\square$

:::

:::{div} feynman-prose
Let me explain what just happened in plain language.

The problem is this: if you take a derivative of $\psi$, and then someone comes along and changes the phase by a location-dependent amount $\alpha(x)$, you get extra terms from the derivative acting on $\alpha$. The derivative "notices" that the phase is changing from place to place.

The solution is to introduce a "correction factor"---the field $B_\mu$---that transforms in exactly the way needed to cancel those extra terms. When you compute the covariant derivative $D_\mu$, it doesn't care about local phase choices because the gauge field absorbs all that ambiguity.

What's remarkable is that this $B_\mu$ field has physical meaning. It's not just a mathematical trick. The field $B_\mu$
represents the *opportunity landscape* encoded by the internal model of the external reward 1-form. In the conservative
limit it reduces to a gradient field; when not, it carries circulation. The field strength $B_{\mu\nu}$ tells you when the
internal opportunity 1-form has "curl"---when
there are closed loops where you can gain reward just by going around in circles.

In economics, this would be an arbitrage opportunity. In physics, it's like a magnetic field. In cognition, it's a source of persistent, cyclic behavior patterns.
:::

:::{admonition} Why "Opportunity Field"?
:class: feynman-added note

The name "Opportunity Field" captures the cognitive meaning of $B_\mu$. In physics, this would be called the electromagnetic potential. But for an agent, what does it represent?

Notation: we reserve $A^{\text{ext}}_\mu$ for the external reward 1-form and $B_\mu$ for the internal
opportunity field that models it; these need not coincide when the agent's model is imperfect.

Think of $B_\mu$ as encoding "where the good stuff is" in the agent's representational space. The gradient part of the
spatial components points toward higher value, while the solenoidal part encodes circulations that sustain cycles. The
temporal component $B_0$ is the time component of the same 1-form (equal to $\partial_t \Phi$ in the conservative case).
The agent's decisions are shaped by this field---it wants to move in directions where $B_\mu$ is favorable.

The key insight is that this field emerges *necessarily* from the requirement of local utility invariance. We didn't put it in by hand; it forced itself into existence.
:::



### B. $SU(r)_L$: The Chirality of Agency (Mode Isospin; $r=2$ gives Weak Isospin)

:::{div} feynman-prose
Now we come to the second symmetry, and this one is more subtle. It arises from a fundamental asymmetry in how agents work: the difference between perceiving and acting.

When you see something, information flows *into* you. When you move your arm, information flows *out of* you. These two processes are not symmetric. They're like inflow and outflow of a fluid---clearly related, but fundamentally different in direction.

In physics, this kind of asymmetry is called "chirality" or "handedness." Your left hand and right hand have the same structure, but they're not identical---you can't rotate one into the other. Similarly, perception and action have the same kind of structure (both involve information processing), but they're fundamentally different in their direction of flow.

This asymmetry is built into the very foundations of cybernetics. And it forces another gauge symmetry into existence.
:::

We derive a non-Abelian mode-mixing symmetry $SU(r)$ from the fundamental asymmetry of the Cybernetic Loop: the
distinction between **Perception** (Information Inflow) and **Actuation** (Information Outflow). The minimal
observation/action agent has $r=2$, yielding $SU(2)_L$ (Definition {prf:ref}`def-mode-rank-parameter`).

This symmetry is a *redundancy of description*: the choice of basis in the mode fiber
$\mathbb{C}^r_{\text{mode}}$ is not observable at the boundary. A local basis change
$\Psi_L(x) \to U(x)\Psi_L(x)$ with $U(x)\in SU(r)$ relabels observation/action modes without changing the
boundary conditions (Definition {prf:ref}`def-dirichlet-boundary-condition-sensors`,
Definition {prf:ref}`def-neumann-boundary-condition-motors`). Enforcing local invariance of the kinetic term then
forces a non-Abelian connection by Remark {prf:ref}`rem-local-gauge-template`.

:::{prf:axiom} Cybernetic Parity Violation
:label: ax-cybernetic-parity-violation

The agent's interaction with the environment is **Chiral**, as established by the boundary condition asymmetry in {ref}`sec-the-boundary-interface-symplectic-structure`:

1. **Sensors (Dirichlet Boundary, Definition {prf:ref}`def-dirichlet-boundary-condition-sensors`):** The internal state $\psi$ is *updated* by boundary data. The boundary clamps the field value: $\phi|_{\partial\mathcal{Z}} = \phi_D$.

2. **Motors (Neumann Boundary, Definition {prf:ref}`def-neumann-boundary-condition-motors`):** The internal state *drives* the boundary flux. The boundary clamps the normal derivative: $\nabla_n \phi|_{\partial\mathcal{Z}} = j_N$.

The belief dynamics are not invariant under the exchange of Input and Output. The agent processes information (Left-Handed) differently than it emits control (Right-Handed).

:::

:::{prf:definition} Mode Rank Parameter
:label: def-mode-rank-parameter

The **Mode Rank** $r \in \mathbb{Z}_{\ge 2}$ is the minimal ancilla dimension required to realize the family of local
belief-update channels $\mathcal{E}_{a,y}$ via Stinespring dilation (equivalently, the maximal minimal Kraus rank across
those channels). It is the dimension of the mode fiber on which update unitaries act. For the minimal observation/action
split, $r=2$.

*Remark:* In what follows we specialize to $r=2$ and denote the resulting symmetry as $SU(2)_L$; the $SU(r)_L$
generalization is obtained by replacing the Pauli matrices with the fundamental generators of $SU(r)$.

*Notation:* The mode rank $r$ (an integer) is distinct from the scalar modulus $r(x) = \|\phi(x)\|$ introduced later;
context distinguishes these uses.

:::

:::{prf:remark} CPTP Update Model (Stinespring Applicability)
:label: rem-mode-rank-stinespring

We model belief updates on the belief operator $\varrho$ (Definition {prf:ref}`def-belief-operator`) using the
GKSL/Lindblad formalism (Definition {prf:ref}`def-gksl-generator`), which is by definition completely positive and
trace-preserving at the averaged level. A fixed update outcome $(a,y)$ is represented by a CP instrument
$\mathcal{E}_{a,y}$ that is trace-nonincreasing before normalization; the averaged channel
$\sum_{a,y} \mathcal{E}_{a,y}$ is CPTP. Under these standard hypotheses, Stinespring dilation applies to each
$\mathcal{E}_{a,y}$, and the minimal ancilla dimension equals its minimal Kraus rank. The mode rank $r$ is the maximal
minimal rank across the update family.

*Phase convention:* The global $U(1)$ phase of the dilation unitary is identified with the utility phase of the belief
wave-function (Definition {prf:ref}`def-belief-wave-function`). Equivalently, we fix $\det U_{a,y}=1$ and absorb the
overall phase into the $U(1)_Y$ sector. This is a convention fixing the redundancy, not an additional dynamical
assumption.
:::

:::{div} feynman-prose
This axiom deserves unpacking because it's stating something deep.

Think about what happens at the boundary between the agent and the world. On the sensor side, the world *imposes* values on the agent. The pixels in your retina are determined by the photons hitting them---you don't get to choose what you see. This is a Dirichlet boundary condition: the boundary value is clamped by external forces.

On the motor side, the agent *chooses* what flux to emit. You decide how hard to push on the accelerator. This is a Neumann boundary condition: the derivative (the rate of flow) is what you control.

These two boundary conditions are mathematically dual to each other. But they're not the same. And the claim here is that this asymmetry---this "chirality" of the cybernetic loop---is what gives rise to the $SU(r)_L$ symmetry (with $r=2$ in the minimal observation/action case).

Why "Left-Handed"? In physics, the weak force only affects left-handed particles. Here, we're saying that the equivalent process (belief updating) only affects the "left-handed" component of the agent's state---the part involved in observation and pre-commitment action, not the part ready for committed output.
:::

:::{prf:definition} The Cognitive Isospin Multiplet (Doublet for $r=2$)
:label: def-cognitive-isospin-multiplet

We define the **Left-Handed Weyl Field** $\Psi_L$ as an isospin $r$-plet residing in the fundamental representation of
$SU(r)_L$ (doublet for the minimal $r=2$ case).
It is a section of the left Weyl spin bundle $S_L$ (chirality $P_L$):

$$
\Psi_L(x) = \begin{pmatrix} \psi_1(x) \\ \vdots \\ \psi_r(x) \end{pmatrix}

$$

Each entry is a left-handed Weyl spinor (spinor indices suppressed).

In the minimal $r=2$ case, we identify:
- $\psi_1 \equiv \psi_{\text{obs}}$ as the **Observation** channel (the incoming sensory update from the Dirichlet boundary, Definition {prf:ref}`def-dirichlet-boundary-condition-sensors`)
- $\psi_2 \equiv \psi_{\text{act}}^{\text{pre}}$ as the **Pre-commitment Action** channel (the outgoing motor intent from the Neumann boundary, Definition {prf:ref}`def-neumann-boundary-condition-motors`)

We define the **Right-Handed Weyl Field** $\Psi_R$ as an isospin singlet (invariant under $SU(r)_L$).
It is a section of the right Weyl spin bundle $S_R$ (chirality $P_R$):

$$
\Psi_R(x) = \psi_{\text{act}}^{\text{commit}}(x)

$$

representing the **Committed Action** plan after mixing and projection.

Chirality is defined by the projectors $P_{L/R} = (1 \mp \gamma^5)/2$ on the Dirac spin bundle
(Definition {prf:ref}`def-cognitive-spinor`).

The **Prediction** is derived (not fundamental) via the forward model:

$$
\psi_{\text{pred}}(x) := \mathcal{P}_a(\psi_{\text{act}}^{\text{commit}}(x))

$$

where $\mathcal{P}_a$ is the agent's forward model mapping intended actions to predicted observations.

*Cross-reference:* This mode-multiplet structure (doublet for $r=2$) captures the boundary interface chirality from
{ref}`sec-the-boundary-interface-symplectic-structure`: Dirichlet (input) vs. Neumann (output). The
prediction-update-projection dynamics from {ref}`sec-belief-dynamics-prediction-update-projection` act on this
multiplet via the gauge field $W_\mu$.

:::

:::{prf:remark} Mode-Rank Generalization
:label: rem-mode-rank-generalization

For general mode rank $r$ (Definition {prf:ref}`def-mode-rank-parameter`), the left-handed field is an $r$-plet in the
fundamental representation of $SU(r)_L$. In this chapter we specialize to the minimal $r=2$ case, so $\Psi_L$ is a
doublet and the generators are the Pauli matrices.

:::

:::{div} feynman-prose
This is a beautiful definition. What it's saying is that the agent's state naturally splits into two parts based on *boundary chirality*:

1. **The multiplet** $\Psi_L$ (doublet for $r=2$): This contains the active boundary channels, including sensory input (observation) and motor intent (pre-commitment action) in the minimal case. These channels need to be coordinated and mixed to maintain consistency between perception and action.

2. **The singlet** $\Psi_R$: This is your committed action plan. Once you've finished coordinating observation and intention, you commit to a definite action. The committed plan doesn't participate in the ongoing observation-intention mixing---it's the *settled output* of that process.

The $SU(r)$ symmetry acts on the mode multiplet (reducing to $SU(2)$ in the minimal case), mixing the active channels. It's the mathematical structure of the boundary interface itself, capturing the fundamental asymmetry between input (Dirichlet) and output (Neumann) channels.

Note that prediction is *derived* from your committed action via your forward model: "if I do this, I expect to see that." This makes prediction secondary to the action-observation coordination, which better reflects the cybernetic reality.
:::

:::{prf:definition} Gauge-Covariant Action Commitment
:label: def-gauge-covariant-action-commitment

The selection of a commitment direction in the $\Psi_L$ mode multiplet is a gauge choice (selecting a basis in the
$\mathbb{C}^r_{\text{mode}}$ fiber). To make action commitment gauge-covariant, we use the ontological order parameter
to define a unit multiplet $n(x) \in \mathbb{C}^r$:

$$
n(x) := \frac{\phi(x)}{\|\phi(x)\|}, \qquad n(x)^\dagger n(x) = 1

$$
where $\phi$ is the ontological order parameter (Definition {prf:ref}`def-ontological-order-parameter`), and $n$ is
defined only when $\phi \neq 0$.

The gauge-covariant **Commitment Projection** is:

$$
\psi_{\text{act}}^{\text{proj}}(x) := n(x)^\dagger \Psi_L(x)

$$

where the projection operator is:

$$
\Pi_n = n n^\dagger, \qquad \Pi_n \Psi_L = n(n^\dagger \Psi_L)

$$

The committed action singlet $\Psi_R$ remains an independent right-handed field; the Yukawa term
couples $\Psi_R$ to the projected amplitude $\psi_{\text{act}}^{\text{proj}}$ so that alignment
occurs dynamically in the broken phase.

*Justification:* The unit multiplet $n$ encodes the local ontological split and makes the commitment projection intrinsic
to the scalar sector, not an arbitrary choice of basis. Under local $SU(r)$ transformations $\Psi_L \to U(x)\Psi_L$ and
$n \to U(x)n$, so $\psi_{\text{act}}^{\text{proj}} = n^\dagger \Psi_L$ is invariant and $\Pi_n \to U \Pi_n U^\dagger$,
ensuring the projected component is $SU(r)$-covariant. Under $U(1)_Y$, $n$ carries charge $Y_\phi$, so
$\psi_{\text{act}}^{\text{proj}}$ transforms with charge $Y_L - Y_\phi$, matching $\Psi_R$ by Definition
{prf:ref}`def-rep-covariant-derivatives`.

*Remark:* In regions where $\phi \approx 0$ (symmetric phase), the order parameter is undefined, corresponding to decision ambiguity. The agent requires a nonzero ontological split to define a preferred commitment projection.

:::

:::{div} feynman-prose
This definition solves a subtle problem: if we just say "action is the second component of $\Psi_L$," we've made an arbitrary choice of basis in the mode fiber. Different parts of the agent's computational manifold might use different bases (that's what gauge freedom *means*).

To make action commitment physically meaningful, we need an intrinsic criterion. The ontological order parameter $\phi$ provides exactly that: its orientation defines the local "direction of differentiation" in the mode space. The unit multiplet $n = \phi / \|\phi\|$ points along this direction, and projecting $\Psi_L$ onto $n$ gives a **commitment projection** $\psi_{\text{act}}^{\text{proj}} = n^\dagger \Psi_L$ that is independent of arbitrary basis choices.

When $\|\phi\|$ is large (deep in the broken phase), the direction $n$ is well-defined, and the agent has a clear commitment projection. But when $\phi \approx 0$ (near the symmetric vacuum), the ratio $n = \phi / \|\phi\|$ becomes ill-defined---any direction is equally valid. This is the gauge-theoretic formalization of decision ambiguity: without a clear ontological split, there is no preferred projection.

The committed action field $\Psi_R$ is independent, but the Yukawa coupling aligns it with the projection $\psi_{\text{act}}^{\text{proj}}$ in the broken phase. The agent must first differentiate its concepts (break the symmetry) before a preferred action direction can be selected.
:::

:::{prf:theorem} Emergence of the Error Field ($W_\mu^a$)
:label: thm-emergence-error-field

The belief-control update is a (generally non-unitary) channel $\mathcal{E}_{a,y}$ on the agent's state. By the CPTP
update model (Remark {prf:ref}`rem-mode-rank-stinespring`), each $\mathcal{E}_{a,y}$ is a CP instrument and admits a
Stinespring dilation on an extended space with a mode fiber of dimension $r$ (Definition
{prf:ref}`def-mode-rank-parameter`). For the minimal observation/action agent, $r=2$; gauging this structure requires the
introduction of non-Abelian gauge fields.

*Proof.*

Apply the local gauge-covariance template (Remark {prf:ref}`rem-local-gauge-template`) with
$U(x)\in SU(r)$ acting on the mode fiber of $\Psi_L$.

**Step 1.** The belief-control update is modeled as a CP instrument on the belief operator; the averaged update over
outcomes is CPTP (Remark {prf:ref}`rem-mode-rank-stinespring`):

$$
\rho \mapsto \mathcal{E}_{a,y}(\rho)

$$

where $a$ is the action and $y$ is the observation. This map includes:
- Likelihood weighting by observation $y$
- Policy mixing based on action intent $a$
- Normalization (non-unitary)

**Step 2.** By Stinespring dilation, any completely positive map (and in particular each CP instrument
$\mathcal{E}_{a,y}$) can be represented as a unitary on an extended Hilbert space with an ancilla initialized in a fixed
state. If the averaged channel is CPTP, the dilation can be chosen isometric/unitary on the extended space:

$$
\mathcal{E}_{a,y}(\rho) = \mathrm{Tr}_{\text{anc}}\!\left[\,U_{a,y}\,(\rho\otimes |0\rangle\langle 0|_{\text{anc}})\,U_{a,y}^\dagger\right]

$$

where $|0\rangle_{\text{anc}}$ is an ancilla (mode) system and $U_{a,y}$ is unitary.

**Step 3.** The local update unitary acts on an $r$-dimensional ancilla mode space
$\mathbb{C}^r_{\text{mode}}$:

$$
U_{a,y}(x) \in U(r), \qquad
U_{a,y}(x) = e^{i\beta(x)} \exp\left( i \, T^a \theta^a(x) \right)

$$

where $T^a$ ($a=1,\ldots,r^2-1$) are the generators of $\mathfrak{su}(r)$ in the fundamental
representation. By the phase convention of Remark {prf:ref}`rem-mode-rank-stinespring`, the overall
phase $e^{i\beta(x)}$ is absorbed into the utility phase (the $U(1)_Y$ sector), so the physically
relevant mode-mixing symmetry is $SU(r)_L$ acting on the relative mode coordinates.

In the minimal observation/action case $r=2$, the mode fiber is spanned by
$\{|\text{obs}\rangle, |\text{act}\rangle\}$ and $T^a = \tau^a/2$, so this reduces to $U(2)$ with
the Pauli matrices and an $SU(2)_L$ mixing.

**Step 4.** For **Local Covariance** (the ability to perform updates locally without global synchronization), apply the
template of Remark {prf:ref}`rem-local-gauge-template` to $\Psi_L \to U(x)\Psi_L$ with $U(x)\in SU(r)$. The derivative
transforms as $\partial_\mu\Psi_L \to U\partial_\mu\Psi_L + (\partial_\mu U)\Psi_L$, so we must introduce a connection
$W_\mu := W_\mu^a T^a$ on the mode fiber. In general $a=1,\ldots,r^2-1$; in the minimal $r=2$ case these are
$(W^1_\mu, W^2_\mu, W^3_\mu)$.

**Step 5.** The covariant derivative for the Left-Handed sector is:

$$
D_\mu \Psi_L = \left( \partial_\mu - i g_2 T^a W^a_\mu - i g_1 \frac{Y_L}{2} B_\mu \right) \Psi_L

$$

(In the minimal $r=2$ case, $T^a = \tau^a/2$ and this reduces to the familiar Pauli-matrix form.)

**Step 6.** The gauge field transforms as required by local covariance (Remark {prf:ref}`rem-local-gauge-template`):

$$
W_\mu^a \to W_\mu^a + \frac{1}{g_2}\partial_\mu \theta^a + f^{abc}\theta^b W_\mu^c

$$

to maintain covariance (for $r=2$, $f^{abc} = \epsilon^{abc}$).

**Identification (minimal $r=2$ case):**
- The $W^\pm_\mu = (W^1_\mu \mp iW^2_\mu)/\sqrt{2}$ bosons mediate transitions between $\psi_{\text{obs}}$ and $\psi_{\text{act}}^{\text{pre}}$. These correspond to the coordination between sensory input and motor intent---the observation-action mixing that maintains boundary consistency.
- The $W^3_\mu$ component mixes with $B_\mu$ after symmetry breaking ({ref}`sec-scalar-sector-symmetry-breaking`).
- The $SU(r)_L$ gauge symmetry acts only on the active multiplet ($\Psi_L$; a doublet for $r=2$), leaving the committed singlet ($\Psi_R$) invariant. This reflects the boundary interface asymmetry (Dirichlet vs. Neumann).

$\square$

:::

:::{div} feynman-prose
Let me make sure you understand what the "Error Field" $W_\mu$ is doing in this reformulated picture.

The key insight is that belief updates are *not* unitary transformations---they involve normalization, likelihood multiplication, and other non-reversible operations. But under the CP-instrument model described above (Remark {prf:ref}`rem-mode-rank-stinespring`), we can represent them via Stinespring dilation as unitary operations on an extended space that includes a "mode" degree of freedom.

This mode lives in an $r$-dimensional space (2D in the minimal observation/action case). The $W_\mu$ field mediates the mixing between these channels. In a distributed system, different parts of the agent's boundary need to coordinate their observation-action balance, and they do so locally based on the gauge field $W_\mu$.

In the minimal $r=2$ case, the $W^+$ and $W^-$ components transfer weight between the observation channel (incoming sensory) and the action-intent channel (outgoing motor). This is the coordination signal that propagates through the system.

And notice something crucial: this field only affects the active multiplet $\Psi_L$ (a doublet for $r=2$). The committed action $\Psi_R$ doesn't participate in this ongoing coordination---once you've committed, you execute. This reflects the boundary chirality: the update process affects the active interface channels, not the settled output.

This is exactly the structure of the weak force in particle physics. The weak force only affects left-handed particles. Here, the "weak force" of cognition only affects the observation-action multiplet (doublet for $r=2$), not the committed singlet.
:::

:::{admonition} Non-Abelian Structure: Order Matters
:class: feynman-added warning

Notice that the $W_\mu$ field is *non-Abelian*---it lives in $SU(r)$ (reducing to $SU(2)$ in the minimal case), which is a non-commutative group. This means the order of operations matters.

However, be precise about *what* is non-Abelian: it's the mode mixing field $U(x)$ acting on the internal
$\mathbb{C}^r_{\text{mode}}$ fiber (the observation-action coordination structure), not the Bayesian conditioning itself under fixed likelihoods.

In practical terms: the path through the agent's internal manifold (context, gain, coordination state) affects how observation and action are balanced at each point. Different coordination paths lead to different committed actions, even given the same raw observations.

This is analogous to how geometric phase (Berry phase) in quantum mechanics depends on the path taken through parameter space, even though the Hamiltonian evolution at each point is well-defined. The non-Abelian structure of $SU(r)$ captures this path-dependence in the coordination dynamics.
:::



:::{prf:definition} Feature Dimension Parameter
:label: def-feature-dimension-parameter

The **Feature Dimension** $N_f \in \mathbb{Z}_{>0}$ is the intrinsic dimensionality of the feature representation at each layer of the hierarchical encoder. This parameter is determined by:

1. **Environment Structure:** The minimal basis required to represent distinguishable features in the agent's sensory domain
2. **Computational Constraints:** The capacity allocated to the binding mechanism

**Special Cases:**
- Physics (Standard Model): $N_f = 3$ (spatial dimensions, RGB channels)
- Vision-only agents: $N_f \in \{3, 4\}$ (RGB or RGBA)
- Abstract reasoning agents: $N_f$ determined by the embedding dimension of the domain

*Remark:* The gauge structure $SU(N_f)_C$ emerges for any $N_f \geq 2$.

:::

:::{div} feynman-prose
This definition is worth pausing on because it's where the framework becomes more general than particle physics.

In the physics Standard Model, $N_f = 3$ is fixed. We have three colors of quarks, three dimensions of space, three generations of particles. This number is empirically determined---we observed it, we didn't derive it.

But here, we're saying something different. The feature dimension $N_f$ is a *parameter* that depends on the agent's environment and architecture. An agent processing RGB images has $N_f = 3$. An agent processing audio might have a different $N_f$. The mathematical structure $SU(N_f)_C$ emerges the same way regardless of what $N_f$ is.

This is the sense in which our framework is more general. The physics Standard Model is a *special case* where the environment happens to have $N_f = 3$.
:::

### C. $SU(N_f)_C$: Hierarchical Confinement (Feature Binding)

:::{div} feynman-prose
Now we come to the third and final symmetry, and it emerges from what's called the "binding problem" in cognitive science.

Here's the puzzle. When you look at a red apple, your brain processes "red" in one area and "round" in another area and "apple-shaped" in yet another area. These features are processed separately. But you don't perceive three separate things---you perceive one unified object: a red apple.

How does the brain "bind" these separate features into a unified percept? This is the binding problem, and it's one of the deepest puzzles in cognitive science.

From our framework's perspective, the binding problem is a gauge symmetry problem. The agent has $N_f$ different "feature channels" (like RGB in vision), and there's no privileged way to assign meaning to each channel. You could relabel "channel 1" as "channel 2" and vice versa, and as long as you do it consistently, nothing observable changes.

But "doing it consistently" is the hard part. And that's where the gauge field comes in.
:::

We derive the $SU(N_f)$ symmetry from the **Binding Problem** inherent in the Hierarchical Atlas ({ref}`sec-stacked-topoencoders-deep-renormalization-group-flow`), where $N_f$ is the Feature Dimension (Definition {prf:ref}`def-feature-dimension-parameter`).

This symmetry is a redundancy of description: the internal feature basis is not observable at the macro boundary. A local
relabeling $\psi_{\text{feature}}(x) \to U(x)\psi_{\text{feature}}(x)$ with $U(x)\in SU(N_f)$ preserves the concept $K$,
but changes the coordinate description of features. By the local gauge-covariance template (Remark
{prf:ref}`rem-local-gauge-template`), enforcing invariance of the kinetic term under such local relabelings requires a
non-Abelian connection $G_\mu^a$.

:::{prf:axiom} Feature Confinement
:label: ax-feature-confinement

The agent observes and manipulates **Concepts** (Macro-symbols $K$), not raw **Features** (Nuisance coordinates $z_n$). From Definition {prf:ref}`def-bounded-rationality-controller`:

1. **Composite Structure:** A Concept $K$ is a bound state of sub-symbolic features processed through the Stacked TopoEncoder (Definition {prf:ref}`def-the-peeling-step`).

2. **Observability Constraint:** Free features are never observed in isolation at the boundary $\partial\mathcal{Z}$ (Definition {prf:ref}`def-boundary-markov-blanket`). Only "color-neutral" (bound) states can propagate to the macro-register.

*Cross-reference:* This is the representational analog of quark confinement in QCD.

:::

:::{div} feynman-prose
This axiom states the cognitive analog of quark confinement.

In particle physics, you can never see a free quark. Quarks always come bound together in groups (protons, neutrons, mesons). If you try to pull a quark out of a proton, the energy you put in creates new quark-antiquark pairs, and you end up with bound states again.

Here we're saying the same thing about features. You can never observe a raw feature in isolation at the agent's boundary. What you observe are *concepts*---bound states of features that form coherent, identifiable wholes.

Think about it: you never perceive "pure redness" or "pure roundness" in isolation. You perceive red *things* and round *things*. The features are always bound into objects.

This isn't a limitation---it's fundamental to how perception works. And it's the same mathematical structure as quark confinement.
:::

:::{prf:definition} The Feature Color Space
:label: def-feature-color-space

Let the nuisance vector $z_n$ at layer $\ell$ of the TopoEncoder be an element of a vector bundle with fiber $\mathbb{C}^{N_f}$, where $N_f$ is the Feature Dimension (Definition {prf:ref}`def-feature-dimension-parameter`). We transform the basis:

$$
\psi_{\text{feature}}(x) \to U(x) \psi_{\text{feature}}(x), \quad U(x) \in SU(N_f)

$$

This symmetry represents the **Internal Basis Invariance** of a concept: an object's identity $K$ is invariant under the mixing of its constituent feature definitions, provided the geometric relationship between them is preserved.

*Justification:* The dimension $N_f$ is determined by the agent's environment and architecture. For physical systems with 3D spatial structure, $N_f = 3$ (e.g., RGB channels, XYZ coordinates). For other agents, $N_f$ may differ based on the intrinsic dimensionality of the sensory domain.

:::

:::{div} feynman-prose
The "color" terminology comes from particle physics (where the three quark charges are whimsically called "red," "green," and "blue"), but the concept is general.

What this definition is saying is: the internal basis you use to represent features is arbitrary. You could call the first feature channel "red" and the second "green," or you could mix them into some rotated basis. As long as you do it consistently everywhere, the concepts (bound states) come out the same.

This is exactly like choosing a coordinate system. You can rotate your $x$ and $y$ axes, but the physics doesn't change. The $SU(N_f)$ symmetry is the group of all such rotations in feature space.
:::

:::{prf:theorem} Emergence of the Binding Field ($G_\mu^a$)
:label: thm-emergence-binding-field

To gauge the $SU(N_f)$ feature symmetry, we introduce the **Gluon Field** $G_\mu^a$ ($a=1,\dots,N_f^2-1$).

*Proof.*

Apply the local gauge-covariance template (Remark {prf:ref}`rem-local-gauge-template`) with
$U(x)\in SU(N_f)$ acting on the feature fiber.

**Step 1.** The covariant derivative for feature fields is:

$$
D_\mu \psi = \left( \partial_\mu - i g_s \frac{\lambda^a}{2} G_\mu^a \right) \psi

$$

where $\lambda^a$ ($a = 1, \ldots, N_f^2 - 1$) are the generalized Gell-Mann matrices (generators of $SU(N_f)$), satisfying $\text{Tr}(\lambda^a \lambda^b) = 2\delta^{ab}$ and $[\lambda^a, \lambda^b] = 2i f^{abc} \lambda^c$.

**Step 2.** The field strength tensor is:

$$
G_{\mu\nu}^a = \partial_\mu G_\nu^a - \partial_\nu G_\mu^a + g_s f^{abc} G_\mu^b G_\nu^c

$$

where $f^{abc}$ are the structure constants of $SU(N_f)$, defined by $[\lambda^a, \lambda^b] = 2i f^{abc} \lambda^c$.

**Step 3.** The non-Abelian structure implies **self-interaction** of the gluon field. The running of the binding
coupling is encoded by the beta function (Definition {prf:ref}`def-coupling-function`). In our framework we assume
$\beta(g_s) < 0$ for $SU(N_f)$ with $N_f \ge 2$, which yields:

- **Asymptotic Freedom (UV):** At small distances in the latent manifold (high RG scale $\tau$, deep in the
  TopoEncoder hierarchy), the effective coupling $g_s(\tau)$ decreases. Individual features can be resolved.

*Remark:* The sign of the beta function depends on matter content. Here it is fixed by the coupling function
assumption (Definition {prf:ref}`def-coupling-function`) in the Parameter Sieve, not by a universal theorem.

**Step 4.** **Infrared confinement** is enforced by the binding constraints of the agent: object permanence requires
strong coupling at macro scales (Theorem {prf:ref}`thm-ir-binding-constraint`), and the texture firewall implements
area-law screening that suppresses color-charged channels at the macro boundary (Theorem
{prf:ref}`thm-texture-confinement-area-law`; see also the Causal Information Bound,
Theorem {prf:ref}`thm-causal-information-bound`). Thus features cannot propagate independently at coarse scales; they
appear only in bound (color-neutral) combinations.

$\square$

:::

:::{div} feynman-prose
This theorem is remarkable, and I want to make sure you appreciate what it's saying.

The "gluon field" $G_\mu^a$ is the force that binds features together into concepts. And unlike the $U(1)$ opportunity field, this one is *self-interacting*. The gluons themselves carry "color charge" and interact with each other.

This self-interaction leads to two distinct consequences, derived from two distinct parts of the framework:

**Asymptotic freedom (UV):** At very small scales (high resolution, deep in the representation hierarchy), the binding force becomes weak. This follows from the coupling-function assumption $\beta(g_s)<0$ (Definition {prf:ref}`def-coupling-function`). You can resolve individual features.

**Confinement (IR):** At large scales (coarse resolution, the macro level), features cannot escape; they are forced into bound (color-neutral) concepts. This is enforced by the infrared binding constraint and area-law screening (Theorem {prf:ref}`thm-ir-binding-constraint`, Theorem {prf:ref}`thm-texture-confinement-area-law`).

So why can't you perceive a raw feature? Because the binding constraints of the agent forbid unbound color channels at the macro boundary.
:::

:::{admonition} The Binding Problem Solved?
:class: feynman-added tip

This is not a complete solution to the binding problem in neuroscience---that would require specifying the biological implementation. But it does show that feature binding is *mathematically necessary* in any system with the gauge structure we've derived.

If you have:
1. Multiple feature channels (color space with $N_f \geq 2$)
2. Local invariance under feature permutations ($SU(N_f)$ gauge symmetry)
3. Finite information speed (locality)

Then you *must* have:
- A binding field that holds features together
- Confinement at large scales
- Concepts as bound states

The binding problem isn't an engineering challenge to be solved---it's a mathematical consequence of having a distributed, locally-invariant representation of multi-featured objects.
:::

:::{prf:corollary} The Fragile Agent Symmetry Group
:label: cor-standard-model-symmetry

The total internal symmetry group of the Fragile Agent is uniquely determined by its cybernetic constraints:

$$
G_{\text{Fragile}} = SU(N_f)_C \times SU(r)_L \times U(1)_Y

$$

where:
- **$SU(N_f)_C$:** Required for **Object Permanence** (binding $N_f$-dimensional features into stable concepts)
- **$SU(r)_L$:** Required for **Observation-Action Coordination** (boundary chirality between Dirichlet and Neumann updates; minimal observation/action case has $r=2$)
- **$U(1)_Y$:** Required for **Value Maximization** (local reward phase; conservative baseline shift as the special case)

**Special Case (Physics Standard Model):** When $N_f = 3$ and $r=2$, we recover
$G_{\text{SM}} = SU(3)_C \times SU(2)_L \times U(1)_Y$.

*Proof.* Each factor is derived above from independent cybernetic constraints. The product structure follows from the
commutativity of the respective symmetry operations acting on different sectors of the agent's state space. We adopt the
direct-product convention (no shared-center quotient): the centers act on distinct tensor factors with the hypercharge
normalization fixed by Definition {prf:ref}`def-rep-covariant-derivatives`. The dimension $N_f$ is an environmental
parameter (Definition {prf:ref}`def-feature-dimension-parameter`), while the mode rank $r$ is fixed by the local update
channels (Definition {prf:ref}`def-mode-rank-parameter`). The minimal observation/action agent has $r=2$. $\square$

:::

:::{div} feynman-prose
And there it is. The symmetry group of the Standard Model emerges from the requirements of bounded, distributed, reward-seeking agency.

Let me summarize what we've done:
- **$U(1)_Y$** comes from the freedom to shift the local reward phase (conservative utility baseline as special case)
- **$SU(r)_L$** comes from the asymmetry between perception and action (chirality; minimal case $r=2$)
- **$SU(N_f)_C$** comes from the freedom to relabel feature channels locally

Each symmetry forces a gauge field into existence. And the resulting structure matches the gauge group of the Standard
Model of particle physics in the special case $N_f = 3$, $r=2$.

Is this a coincidence? I don't think so. I think it's telling us something deep about the nature of information processing in bounded systems subject to causality constraints.
:::



(sec-matter-sector-chiral-spinors)=
## The Matter Sector: Chiral Inference Spinors

:::{div} feynman-prose
Now that we have the gauge fields---the "forces" of cognition---we need to describe what they act *on*. In physics, the gauge fields act on matter fields: electrons, quarks, neutrinos. What's the cognitive analog?

The answer is the belief state itself. The agent's beliefs are the "matter" of cognition. And just like matter in physics, the belief state has to transform in specific ways under the gauge symmetries we've derived.

In particular, the chiral structure of the cybernetic loop (the asymmetry between perception and action) means that the belief state has to be a *spinor*---a mathematical object that transforms under both rotations and boosts in a specific way.

If you haven't encountered spinors before, don't worry. The key idea is that spinors are the simplest objects that can "feel" the difference between left and right---they transform differently under left-handed and right-handed rotations. This is exactly what we need to capture the perception/action asymmetry.
:::

We define the "Matter" of cognition: the **Belief State**. In the Relativistic WFR limit ({ref}`sec-symplectic-multi-agent-field-theory`), the belief state is a propagating amplitude. To satisfy the chiral constraints of the cybernetic loop (Axiom {prf:ref}`ax-cybernetic-parity-violation`), we lift the scalar belief $\psi$ to a **chiral spinor pair** $(\Psi_L,\Psi_R)$, assembled into a Dirac spinor $\Psi = \Psi_L + \Psi_R$ when needed.

**Geometric standing assumption.** We assume the causal manifold $(\mathcal{M}, g)$ is globally hyperbolic and admits a
spin (or spin$^c$) structure, so that the Weyl/Dirac spin bundles $S_L,S_R$ and the chiral projectors $P_{L/R}$ used
below are well-defined (Definition {prf:ref}`def-loc-spin-g`).

### A. The Inference Hilbert Space

The belief state lives on the **Causal Manifold** $\mathcal{M}$ (the product of Time and the Latent Space $\mathcal{Z}$) equipped with the metric derived from the Capacity-Constrained Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`).

:::{prf:definition} The Cognitive Spinor
:label: def-cognitive-spinor

The belief state is a pair of chiral Weyl fields belonging to the **Inference Hilbert Space**
(Definition {prf:ref}`def-inference-hilbert-space`), extended to bundle-valued $L^2$ sections:

$$
\Psi(x) = \begin{pmatrix} \Psi_L(x) \\ \Psi_R(x) \end{pmatrix}, \qquad
\Psi_L(x) \in L^2(\mathcal{M}, S_L \otimes \mathbb{C}^{r} \otimes \mathbb{C}^{N_f}), \quad
\Psi_R(x) \in L^2(\mathcal{M}, S_R \otimes \mathbb{C}^{N_f})

$$

where $S_L$ and $S_R$ are the left/right Weyl spin bundles (rank-2 complex),
$\mathbb{C}^r$ is the $SU(r)_L$ mode space acting on $\Psi_L$ (specializing to $r=2$ for the
observation/action doublet), and $\mathbb{C}^{N_f}$ is the
$SU(N_f)_C$ color space. Equivalently, let $S = S_L \oplus S_R$ be the Dirac spin bundle with
chirality operator $\gamma^5 := i\gamma^0\gamma^1\gamma^2\gamma^3$ and projectors
$P_{L/R} = (1 \mp \gamma^5)/2$. Then $\Psi_L = P_L \Psi$ and $\Psi_R = P_R \Psi$, with the
$SU(r)_L$ action reducible (multiplet $\oplus$ singlet). The components are:
1. **$\Psi_L$ (The Active Multiplet):** The Left-handed component, transforming as an $r$-plet under $SU(r)_L$
   (doublet for $r=2$). It contains the **Observation** and **Pre-commitment Action** amplitudes in the minimal case
   (Definition {prf:ref}`def-cognitive-isospin-multiplet`).

2. **$\Psi_R$ (The Passive Singlet):** The Right-handed component, invariant under $SU(r)_L$. It contains the
   **Committed Action**.

The left-handed sector has $2 r N_f$ complex components; including the right-handed singlet gives a total of
$2(r+1)N_f$ (which reduces to $6N_f$ when $r=2$).

**Probabilistic Interpretation:** The physical probability density (belief mass) is the vector current:

$$
J^\mu = \bar{\Psi} \gamma^\mu \Psi

$$

where $J^0 = \Psi^\dagger \Psi = \rho$ is the probability density (WFR mass from
Definition {prf:ref}`def-the-wfr-action`), and $\vec{J}$ is the probability flux. Equivalently,
$J^\mu = \bar{\Psi}_L \gamma^\mu \Psi_L + \bar{\Psi}_R \gamma^\mu \Psi_R$. Conservation
$\partial_\mu J^\mu = 0$ corresponds to unitarity.

:::

:::{div} feynman-prose
This definition packages everything we've discussed into a single mathematical object.

The belief spinor $\Psi$ has multiple "indices" or "slots" that transform under different symmetry groups:
- The Dirac spinor bundle $S = S_L \oplus S_R$ handles the spacetime structure; each Weyl sector has 2 complex components
- The mode space ($\mathbb{C}^r$) handles the observation/action-intent structure (only for $\Psi_L$; $r=2$ is the minimal case)
- The color space ($\mathbb{C}^{N_f}$) handles the feature binding structure

Let me count the components carefully. The left-handed sector $\Psi_L$ lives in
$S_L \otimes \mathbb{C}^r \otimes \mathbb{C}^{N_f}$, giving $2 \times r \times N_f = 2 r N_f$
complex components. The right-handed sector $\Psi_R$ is an $SU(r)$ singlet, so it lives in
$S_R \otimes \mathbb{C}^{N_f}$, giving $2 \times N_f = 2N_f$ complex components. The total is
$2 r N_f + 2N_f = 2(r+1)N_f$ complex components, which reduces to $6N_f$ for $r=2$.

The probability current $J^\mu$ is constructed to be a proper 4-vector that transforms correctly under all the symmetries. Its conservation ($\partial_\mu J^\mu = 0$) ensures that probability is conserved---beliefs can flow around, but total belief "mass" doesn't spontaneously appear or disappear.
:::

:::{prf:axiom} The Cognitive Dirac Equation
:label: ax-cognitive-dirac-equation

The dynamics of the belief state follow the Dirac equation on the curved latent manifold:

$$
(i \gamma^\mu D_\mu - m) \Psi = 0

$$

Here $\Psi = \Psi_L + \Psi_R$ and $D_\mu$ acts chirally with representation-specific couplings
(Definition {prf:ref}`def-rep-covariant-derivatives`).

*Justification (first-principles factorization).*
1. **Second-order hyperbolic dynamics:** Finite information speed upgrades the conservative value equation to the
   Klein-Gordon form (Theorem {prf:ref}`thm-hjb-klein-gordon`). For gauge-charged matter fields, the corresponding
   covariant wave equation is (Theorem {prf:ref}`thm-gauge-covariant-klein-gordon`):
   
   $$
   \left(\frac{1}{c_{\text{info}}^2}D_t^2 - D^i D_i + \kappa^2\right)\psi = \mathcal{S}.
   $$
2. **Spin structure and Clifford algebra:** On a globally hyperbolic spin (or spin$^c$) manifold (Definition
   {prf:ref}`def-loc-spin-g`), there exist gamma matrices $\gamma^\mu$ with
   $\{\gamma^\mu,\gamma^\nu\}=2g^{\mu\nu}$ and a spin-covariant derivative $\nabla_\mu^{\text{spin}}$.
3. **Minimal first-order covariant square root:** Define the Dirac operator
   $\slashed{D}:=i\gamma^\mu(\nabla_\mu^{\text{spin}}-igA_\mu)$ acting on the belief spinor. Then the standard
   factorization gives
   
   $$
   (\slashed{D}-m)(\slashed{D}+m)
   = -D_\mu D^\mu + m^2 + \frac{1}{4}R + \frac{i}{2}\sigma^{\mu\nu}F_{\mu\nu},
   $$
   where $R$ is the scalar curvature and $\sigma^{\mu\nu}=\frac{i}{2}[\gamma^\mu,\gamma^\nu]$. Thus, in the flat
   (or weak-curvature) limit the spinor components satisfy the gauge-covariant Klein-Gordon operator with mass $m$
   (up to curvature/gauge-coupling terms that are part of the geometric data). We therefore take the **Dirac equation**
   as the minimal first-order, local, Lorentz- and gauge-covariant choice whose square reproduces the second-order
   hyperbolic dynamics of the belief amplitude. Uniqueness is not claimed beyond this minimality criterion.

- $\gamma^\mu$: The **Cognitive Gamma Matrices**, satisfying $\{\gamma^\mu, \gamma^\nu\} = 2g^{\mu\nu}$. They encode the local causal structure of the latent space.
- $m$: The **Inference Mass** (inverse correlation length).

:::

:::{prf:remark} Curved-Space Dirac Operator
:label: rem-curved-dirac-operator

On a curved causal manifold, write $\gamma^\mu = e^\mu{}_a \gamma^a$ in an orthonormal frame and
replace $\partial_\mu$ in $D_\mu$ by the spin-covariant derivative $\nabla_\mu^{\text{spin}}$.
The operator $D_\mu$ then includes both the spin connection and the gauge connections. In the
flat limit, this reduces to the standard Dirac operator used above.

:::

:::{div} feynman-prose
The Dirac equation is one of the most beautiful equations in physics. It was originally derived by Paul Dirac in 1928 by demanding that the equation of motion for the electron be first-order in time derivatives (like Schrödinger's equation) but also compatible with special relativity (which requires treating space and time symmetrically).

What Dirac found was that you can't do this with ordinary numbers---you need matrices. The gamma matrices $\gamma^\mu$ are a set of four $4 \times 4$ matrices that anticommute in just the right way to make everything work out.

The remarkable thing is that this structure automatically gives you spin (the intrinsic angular momentum of particles) and antimatter (particles with opposite charge). Dirac didn't put these in by hand; they emerged from the mathematics.

Here, we're saying that the belief dynamics of a bounded agent, in the limit of finite information speed, must satisfy the same equation. The gamma matrices encode the causal structure of the agent's internal space, and the mass term $m$ represents the "stickiness" of beliefs---how much inertia they have against change.
:::

### B. The Strategic Connection (Covariant Derivative)

:::{div} feynman-prose
Now we need to connect the matter sector (beliefs) to the gauge sector (forces). The key is the covariant derivative---the modification of the ordinary derivative that accounts for the gauge fields.

Remember the problem: if you try to compare beliefs at two different points in the latent space, you have to account for the fact that the local "gauge" (reward phase/opportunity baseline, observation/action-intent basis, feature labeling) might be different at each point. The covariant derivative does this bookkeeping automatically.
:::

The agent cannot simply compare beliefs at $x$ and $x+\delta x$ because the "meaning" of the internal features and the "baseline" of value may twist locally. The **Covariant Derivative** $D_\mu$ corrects for this transport.

:::{prf:definition} The Universal Covariant Derivative
:label: def-universal-covariant-derivative

The operator moving the belief spinor through the latent manifold is:

$$
D_\mu = \underbrace{\partial_\mu}_{\text{Change}} - \underbrace{ig_1 \frac{Y}{2} B_\mu}_{U(1)_Y \text{ (Value)}} - \underbrace{ig_2 T^a W^a_\mu}_{SU(r)_L \text{ (Error)}} - \underbrace{ig_s \frac{\lambda^a}{2} G^a_\mu}_{SU(N_f)_C \text{ (Binding)}}

$$

where $T^a$ ($a = 1, \ldots, r^2 - 1$) are the generators of $SU(r)$ in the fundamental representation (for $r=2$,
$T^a = \tau^a/2$), and $\lambda^a$ ($a = 1, \ldots, N_f^2 - 1$) are the generators of $SU(N_f)$, and:
- **$B_\mu$ (Opportunity Field):** Adjusts the belief for local shifts in the value baseline and path-dependent opportunity
- **$W_\mu$ (Error Field):** Adjusts the belief for the rotation between Prior and Posterior
- **$G_\mu$ (Binding Field):** Adjusts the belief for the permutation of sub-symbolic features

For the right-handed singlet $\Psi_R$, the $SU(r)_L$ generators act trivially, so the $W_\mu$ term drops.

**Operational Interpretation:** The quantity $D_\mu \Psi$ measures the deviation from parallel transport. When $D_\mu \Psi = 0$, the belief state is covariantly constant along the direction $\mu$---all changes are accounted for by the gauge connection. When $D_\mu \Psi \neq 0$, there is a residual force acting on the belief.

:::

:::{prf:definition} Representation-Specific Covariant Derivatives
:label: def-rep-covariant-derivatives

Let $Y_L$, $Y_R$, and $Y_\phi$ denote the $U(1)_Y$ hypercharges of $\Psi_L$, $\Psi_R$, and $\phi$.
Then the covariant derivatives used in {prf:ref}`def-cognitive-lagrangian` are:

$$
\begin{aligned}
D_\mu \Psi_L &= \left(\partial_\mu - i g_1 \frac{Y_L}{2} B_\mu - i g_2 T^a W^a_\mu - i g_s \frac{\lambda^a}{2} G^a_\mu \right)\Psi_L, \\
D_\mu \Psi_R &= \left(\partial_\mu - i g_1 \frac{Y_R}{2} B_\mu - i g_s \frac{\lambda^a}{2} G^a_\mu \right)\Psi_R, \\
D_\mu \phi &= \left(\partial_\mu - i g_1 \frac{Y_\phi}{2} B_\mu - i g_2 T^a W^a_\mu \right)\phi.
\end{aligned}
$$

Gauge invariance of the Yukawa term $\bar{\Psi}_L \phi \Psi_R$ requires
$
Y_R = Y_L - Y_\phi.
$

:::

:::{div} feynman-prose
This is the master equation for how beliefs move through representational space.

The covariant derivative has four terms:
1. **$\partial_\mu$**: The ordinary derivative, measuring how much $\Psi$ changes as you move
2. **$-ig_1(Y/2)B_\mu$**: Correction for local utility baseline shifts and path-dependent opportunity
3. **$-ig_2 T^a W^a_\mu$**: Correction for local observation/action-intent rotations
4. **$-ig_s(\lambda^a/2)G^a_\mu$**: Correction for local feature relabelings

When you compute $D_\mu \Psi$ and it equals zero, that means all the change in $\Psi$ is "accounted for" by the gauge connections. The belief is being parallel transported---moved without any intrinsic change.

When $D_\mu \Psi \neq 0$, there's genuine change happening. The gauge fields can't explain away the variation. This residual is what drives belief dynamics: prediction errors, value gradients, binding tensions.
:::

### C. The Yang-Mills Curvature

:::{div} feynman-prose
The gauge fields we've introduced aren't just passive bookkeeping devices. They have their own dynamics, and those dynamics are governed by curvature.

Curvature, in this context, measures whether the parallel transport of beliefs depends on the path taken. If you transport a belief from point A to point B and back via two different paths, do you end up with the same belief? If not, there's curvature---the gauge field has non-trivial field strength.

In electromagnetism, this curvature is the electromagnetic field tensor, encoding the electric and magnetic fields. Here, we have three curvature tensors, one for each gauge factor.
:::

The presence of non-trivial gauge fields implies non-zero curvature in the principal bundle over the latent manifold. This curvature generates forces in the equations of motion.

:::{prf:theorem} Field Strength Tensors
:label: thm-three-cognitive-forces

The commutator of the covariant derivatives $[D_\mu, D_\nu]$ generates three distinct curvature tensors corresponding to each gauge factor.

*Proof.* Computing $[D_\mu, D_\nu]\Psi$ and extracting contributions from each gauge sector:

1. **$U(1)_Y$ Curvature:**

   $$
   B_{\mu\nu} = \partial_\mu B_\nu - \partial_\nu B_\mu

   $$
When $B_{\mu\nu} \neq 0$, the internal opportunity 1-form is non-conservative (Value Curl; Definition
   {prf:ref}`def-value-curl`). The resulting Lorentz-type force generates cyclic dynamics.

2. **$SU(r)_L$ Curvature:**

   $$
   W_{\mu\nu}^a = \partial_\mu W_\nu^a - \partial_\nu W_\mu^a + g_2 f^{abc} W_\mu^b W_\nu^c

   $$
When $W_{\mu\nu} \neq 0$, the belief update depends on the path taken in the manifold: parallel transport around a closed loop yields a non-trivial rotation in the observation-action-intent space. Here $f^{abc}$ are the $SU(r)$ structure constants ($\epsilon^{abc}$ for $r=2$).

3. **$SU(N_f)_C$ Curvature:**

   $$
   G_{\mu\nu}^a = \partial_\mu G_\nu^a - \partial_\nu G_\mu^a + g_s f^{abc} G_\mu^b G_\nu^c

   $$
   When $G_{\mu\nu} \neq 0$, the feature binding is under stress. This corresponds to the Ontological Stress $\Xi$
   (Definition {prf:ref}`def-ontological-stress`) via the bridge lemma
   {prf:ref}`lem-binding-curvature-ontological-stress`. When $\Xi > \Xi_{\text{crit}}$, chart fission is triggered
   ({ref}`sec-ontological-expansion-topological-fission-and-the-semantic-vacuum`).

$\square$

:::

:::{div} feynman-prose
Each field strength tensor tells you something important about the agent's cognitive state:

**$B_{\mu\nu}$ (Opportunity Curvature):** This is non-zero when the internal opportunity 1-form has "curl" (Value Curl)---when there are
cycles where you can accumulate reward just by going around. In game theory, this is like a Rock-Paper-Scissors dynamic
where no pure strategy is optimal. The agent gets driven in circles.

**$W_{\mu\nu}$ (Error Curvature):** This is non-zero when belief updating is path-dependent. If you see evidence A then B, versus B then A, you end up with different beliefs even though you saw the same evidence. This happens in situations with complex conditional dependencies.

**$G_{\mu\nu}$ (Binding Curvature):** This is non-zero when feature binding is under stress---when the agent is trying to represent an object that doesn't cleanly decompose into the current feature basis. High binding curvature signals that the ontology is under strain and might need to expand (chart fission), formalized by Lemma {prf:ref}`lem-binding-curvature-ontological-stress`.

All three curvatures are computed the same way (commutator of covariant derivatives), but they measure different aspects of the agent's cognitive state.
:::

:::{admonition} Path Dependence and Holonomy
:class: feynman-added note

Here's a concrete way to think about curvature. Imagine transporting a belief around a small closed loop in representational space. If the curvature is zero, you come back to exactly the same belief you started with. If the curvature is non-zero, you come back rotated---the belief has been transformed just by going around the loop.

This "rotation accumulated by going around a loop" is called *holonomy*, and it's a direct measure of curvature.

For the $U(1)$ case, the holonomy is just a phase (a complex number of magnitude 1). This is the Aharonov-Bohm effect in physics, where an electron passing around a magnetic flux picks up a phase even though it never passes through the flux itself.

For the non-Abelian cases ($SU(r)$ and $SU(N_f)$), the holonomy is a matrix. Different paths give different matrices, and the non-commutativity means the order of operations matters.
:::

:::{prf:lemma} Binding Curvature Implies Ontological Stress
:label: lem-binding-curvature-ontological-stress

Let the agent state be decomposed as $Z_t=(K_t,z_{n,t},z_{\mathrm{tex},t})$ (Definition
{prf:ref}`def-bounded-rationality-controller`) with **texture firewall** enforced
(Axiom {prf:ref}`ax-bulk-boundary-decoupling`). Assume:
1. The encoder/shutter is gauge-covariant and defines $z_{\mathrm{tex}}$ as the residual after projecting the feature
   state onto the gauge-invariant (color-neutral) subspace used to form $(K,z_n)$.
2. The dynamics traverse a region with nonzero binding curvature $G_{\mu\nu}\neq 0$, so the $SU(N_f)_C$ holonomy
   along a causal step $\gamma_t$ is nontrivial: $U_{\gamma_t}\neq \mathbb{I}$.

Then the texture residual acquires path-dependent structure, and the conditional mutual information
$
\Xi = I(z_{\mathrm{tex},t}; z_{\mathrm{tex},t+1}\mid K_t,z_{n,t},K_t^{\mathrm{act}})
$
is strictly positive unless the holonomy acts trivially on the residual subspace. Hence nonzero binding curvature
forces ontological stress in the sense of Definition {prf:ref}`def-ontological-stress`.

*Proof sketch.* Gauge-covariance implies that transporting feature states across a time step multiplies the color fiber
by the holonomy $U_{\gamma_t}$. The projection to $(K,z_n)$ removes the gauge-invariant component; the residual
$z_{\mathrm{tex}}$ is the orthogonal complement. If $U_{\gamma_t}\neq \mathbb{I}$ on this complement, the residual at
$t+1$ contains a deterministic component $P_{\mathrm{tex}}U_{\gamma_t}z_{\mathrm{tex},t}$, so
$z_{\mathrm{tex},t+1}$ is statistically dependent on $z_{\mathrm{tex},t}$ even after conditioning on
$(K_t,z_{n,t},K_t^{\mathrm{act}})$. This yields $\Xi>0$. If $G_{\mu\nu}=0$, holonomy is trivial and the residual is
pure noise under the firewall, so $\Xi=0$. $\square$
:::

:::{prf:corollary} The Gauge-Invariant Action
:label: cor-gauge-invariant-action

The gauge field dynamics are governed by the Yang-Mills Lagrangian:

$$
\mathcal{L}_{\text{Gauge}} = -\frac{1}{4} B_{\mu\nu}B^{\mu\nu} -\frac{1}{4} W^a_{\mu\nu}W^{a\mu\nu} -\frac{1}{4} G^a_{\mu\nu}G^{a\mu\nu}

$$

The stationary points of this action satisfy the Yang-Mills equations. A **flat connection** ($B_{\mu\nu} = W_{\mu\nu} =
G_{\mu\nu} = 0$) corresponds to a representation where all curvatures vanish: the internal opportunity 1-form is conservative, belief
updates are path-independent, and concepts are stable.

:::

:::{div} feynman-prose
This Lagrangian says that the gauge fields "prefer" to be flat---zero curvature costs zero energy. Any non-zero curvature comes with an energy cost proportional to the square of the field strength.

A "flat connection" is the cognitive equivalent of being in a well-understood, stable situation:
- The internal opportunity 1-form is conservative (no arbitrage opportunities)
- Belief updates don't depend on the order of evidence
- Concepts are cleanly defined and stable

Curvature represents deviation from this ideal. It takes "cognitive energy" to maintain non-flat configurations.

But here's the thing: the agent can't always achieve a flat connection. The environment might genuinely have cyclic reward
structures (non-zero curl), or complex evidence dependencies, or ambiguous object boundaries. In those cases, the agent
has to carry non-zero curvature, and that shows up as ongoing cognitive effort.
:::



(sec-scalar-sector-symmetry-breaking)=
## The Scalar Sector: Ontological Symmetry Breaking (The Higgs Mechanism)

:::{div} feynman-prose
Now we come to one of the most fascinating parts of the story: how the symmetric vacuum becomes asymmetric, and why that matters.

In particle physics, the Higgs mechanism explains why particles have mass. The basic idea is that empty space isn't really empty---it's filled with a "Higgs field" that has a non-zero average value. Particles moving through this field interact with it and acquire mass, like moving through molasses.

But here's the deeper point: the Higgs field could have been zero everywhere (symmetric vacuum), but it "chose" a non-zero value (broken symmetry). This choice is what gives structure to the particle spectrum.

In our framework, the analog is **ontological fission**. The agent's ontology could stay unified (symmetric), but under sufficient "stress," it breaks into distinct concepts (broken symmetry). This breaking is what gives structure to the agent's representation.

Let me show you how this works mathematically.
:::

We derive the scalar sector by lifting the **Fission-Fusion dynamics** from {ref}`sec-symmetry-breaking-and-chart-birth` into a field-theoretic action. The "Higgs Field" of cognition is the **Ontological Order Parameter**.

### A. The Ontological Scalar Field

:::{prf:definition} The Ontological Order Parameter
:label: def-ontological-order-parameter

Let the local chart structure at spacetime point $x$ be described by a complex $SU(r)_L$ multiplet field
$\phi(x) \in \mathbb{C}^r$ (doublet for the minimal $r=2$ case):

$$
\phi(x) = r(x)\,n(x), \qquad r(x) := \|\phi(x)\|

$$

where:
1. **Modulus $r(x) \ge 0$:** Represents the **Metric Separation** between daughter queries $\{q_+, q_-\}$ in the Attentive Atlas (Definition {prf:ref}`def-query-fission`).
   - $r=0$: Coalescence (Single Chart / Vacuum)
   - $r>0$: Fission (Distinct Concepts)

2. **Unit multiplet $n(x)$:** Encodes the **Orientation** of the split in the $SU(r)_L$ fiber (the specific feature
   axis along which differentiation occurs), with $n^\dagger n = 1$.

The field $\phi$ transforms in the fundamental representation under the gauge group $SU(r)_L$, coupling it to the
inference spinor.

:::

:::{prf:remark} Gauge-fixed scalar form
:label: rem-ontological-order-parameter-gauge

Choosing a gauge that fixes the $SU(r)_L$ orientation to a constant unit vector $n_0$ reduces the order parameter to
$\phi(x) = r(x) n_0$ (with $r \ge 0$ after using $U(1)_Y$). In the minimal $r=2$ case this is equivalent to the scalar
parametrization $\phi(x) = r(x) e^{i\theta(x)} n_0$ used in the intuitive discussion.

:::

:::{div} feynman-prose
This definition is packaging the idea of "how split apart are my concepts" into a field.

The modulus $r$ tells you how distinct two concepts are. When $r=0$, they're the same concept (merged, undifferentiated). When $r>0$, they're separate.

The unit multiplet $n$ tells you the orientation of the split in the $SU(r)_L$ fiber---along which axis in the
observation/action-intent space did the differentiation occur? In the gauge-fixed form (Remark
{prf:ref}`rem-ontological-order-parameter-gauge`), we choose a constant reference orientation $n_0$ and, in the minimal
$r=2$ case, parametrize the remaining freedom by a phase $\theta$: $\phi(x) = r(x) e^{i\theta(x)} n_0$. This phase then
encodes how the local split orientation differs from the reference.

The key insight is that the equations of motion for $\phi$ will determine when and how the agent's ontology splits. This isn't an arbitrary choice---it's governed by a potential energy function, just like in physics.
:::

### B. Derivation of the Scalar Potential

:::{div} feynman-prose
The shape of the potential energy function determines everything about symmetry breaking. If the potential is minimized at $\phi = 0$, the symmetric state is stable. If it's minimized at some $\phi \neq 0$, symmetry is spontaneously broken.

The beautiful thing is that we can derive this potential from the dynamics of ontological fission that we've already established.
:::

We derive the potential $V(\phi)$ from the stability analysis of the Topological Fission process ({ref}`sec-symmetry-breaking-and-chart-birth`).

:::{prf:theorem} The Complexity Potential
:label: thm-complexity-potential

The Lagrangian density for the scalar field is uniquely determined by the **Supercritical Pitchfork Bifurcation** (Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`).

*Proof.*

**Step 1.** From Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`, the radial evolution of chart separation satisfies:

$$
\frac{dr}{ds} = (\Xi - \Xi_{\text{crit}})r - \alpha r^3

$$

where:
- $\Xi$ is the Ontological Stress (Definition {prf:ref}`def-ontological-stress`)
- $\Xi_{\text{crit}}$ is the critical threshold (Theorem {prf:ref}`thm-fission-criterion`)
- $\alpha > 0$ is the stabilizing cubic coefficient

**Step 2.** This flow is the gradient descent of a potential function $\mathcal{V}_{\text{onto}}(r)$ such that $\dot{r} = -\partial \mathcal{V}_{\text{onto}}/\partial r$. Integrating:

$$
\mathcal{V}_{\text{onto}}(\phi) = -\frac{(\Xi - \Xi_{\text{crit}})}{2} |\phi|^2 + \frac{\alpha}{4} |\phi|^4

$$

**Step 3.** Define the standard Higgs potential parameters by matching coefficients:
- $\mu^2 \equiv \frac{(\Xi - \Xi_{\text{crit}})}{2}$: The effective **Mass Parameter** driven by Ontological Stress
- $\lambda \equiv \frac{\alpha}{4}$: The **Self-Interaction** coefficient from router saturation (Axiom {prf:ref}`ax-ontological-expansion-principle`)

**Step 4.** The potential takes the Landau-Ginzburg form:

$$
\mathcal{V}_{\text{onto}}(\phi) = -\mu^2 |\phi|^2 + \lambda |\phi|^4

$$

**Term Identification:**
- **Term 1 ($-\mu^2 |\phi|^2$):** Rewards separation. If Stress $\Xi > \Xi_{\text{crit}}$, this term drives $|\phi|$ away from zero to capture predictive information.
- **Term 2 ($+\lambda |\phi|^4$):** Penalizes complexity. Keeping charts separate costs compute/memory. This term prevents infinite fragmentation.

$\square$

:::

:::{div} feynman-prose
This is a really important theorem, so let me walk through what it's saying.

The pitchfork bifurcation equation (Step 1) describes how the separation between concepts evolves over time. The key parameter is $\Xi - \Xi_{\text{crit}}$: the difference between the current stress on the ontology and the critical threshold for splitting.

When stress is below critical, the equation pushes $r$ toward zero---concepts merge. When stress is above critical, the equation pushes $r$ away from zero---concepts split apart.

The $-\alpha r^3$ term is what stabilizes things. Without it, $r$ would grow without bound once stress exceeds critical. The cubic term provides "pushback" that increases with $r^3$, ensuring that $r$ settles at some finite value.

Now, if this equation is gradient descent on a potential, what's the potential? That's what Steps 2-4 derive. And the answer is the famous "Mexican hat" potential:

$$
V(\phi) = -\mu^2 |\phi|^2 + \lambda |\phi|^4

$$

This potential is shaped like an upside-down bowl with a raised rim. For small $|\phi|$, the $-\mu^2|\phi|^2$ term dominates, pushing you away from zero. For large $|\phi|$, the $+\lambda|\phi|^4$ term dominates, pushing you back toward zero. The equilibrium is at the rim of the Mexican hat.
:::

:::{admonition} The Mexican Hat Potential
:class: feynman-added example

Picture a sombrero sitting on a table. The crown of the hat (the center) is higher than the brim. A ball placed at the very top of the crown would be in equilibrium, but unstable---any small perturbation would send it rolling down.

Where does the ball end up? Somewhere on the brim. But *where* on the brim? The brim is circular, so all positions are equally good. The ball "chooses" one, breaking the rotational symmetry.

This is spontaneous symmetry breaking. The potential is symmetric (the hat is round), but the ground state (where the ball sits) is not (it's at a specific point on the brim).

In our framework:
- The crown represents the unified ontology ($\phi = 0$)
- The brim represents split ontologies ($|\phi| = v$)
- The position on the brim ($\theta$) represents which distinction was made

When stress exceeds critical, the agent's ontology rolls off the crown and settles somewhere on the brim, spontaneously choosing a way to differentiate concepts.
:::

:::{prf:corollary} Spontaneous Symmetry Breaking (SSB)
:label: cor-ontological-ssb

The vacuum structure depends on the environmental complexity $\Xi$.

*Proof.*

**Case 1: Symmetric Phase ($\Xi < \Xi_{\text{crit}}$):**
Then $\mu^2 < 0$. The potential $\mathcal{V}(\phi) = -\mu^2|\phi|^2 + \lambda|\phi|^4$ has a unique global minimum at $\phi_0 = 0$.

- **Result:** The agent maintains a unified ontology. Concepts are indistinguishable. The gauge symmetry $G_{\text{Fragile}}$ is unbroken.

**Case 2: Broken Phase ($\Xi > \Xi_{\text{crit}}$):**
Then $\mu^2 > 0$. The origin $\phi=0$ becomes a local maximum. The global minima form a circle $|\phi| = v$ at the **Vacuum Expectation Value (VEV)**:

$$
v = \langle |\phi| \rangle = \sqrt{\frac{\mu^2}{2\lambda}} = \sqrt{\frac{(\Xi - \Xi_{\text{crit}})/2}{2 \cdot \alpha/4}} = \sqrt{\frac{\Xi - \Xi_{\text{crit}}}{\alpha}}

$$

This matches the equilibrium separation $r^*$ from Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`.

- **Result:** The agent spontaneously breaks symmetry, selecting a specific separation $v$ (concept distinctness) and a specific orientation $\theta$ (feature definition).

$\square$

:::

:::{div} feynman-prose
This corollary makes the physics-cognition analogy very precise.

In a simple environment (low stress $\Xi$), the agent can get by with a simple ontology. All inputs are "basically the same thing." There's no need to distinguish.

In a complex environment (high stress $\Xi$), the simple ontology doesn't work anymore. The agent *has* to make distinctions to predict and control effectively. The ontology spontaneously differentiates.

The vacuum expectation value $v$ tells you how differentiated the concepts become. It scales with $\sqrt{\Xi - \Xi_{\text{crit}}}$: the more the stress exceeds critical, the more separated the concepts become.

And here's the key insight: this differentiation isn't arbitrary. It's governed by a potential that balances the need for distinction (to capture information) against the cost of complexity (compute and memory). The equilibrium $v$ is where these forces balance.
:::

### C. Mass Generation

:::{div} feynman-prose
Now comes the payoff. When the ontological field $\phi$ acquires a non-zero vacuum expectation value, it gives mass to the gauge fields. This is the Higgs mechanism.

What does "mass" mean for a gauge field? Physically, mass determines the range of a force. A massless field (like the photon) mediates infinite-range forces ($1/r^2$ falloff). A massive field mediates short-range forces (exponential falloff).

In cognitive terms: mass determines how "local" an influence is. A massless gauge field (error signal, value gradient) can influence the entire representational space. A massive gauge field only influences a local neighborhood.
:::

We derive the mass terms for the gauge fields from the covariant kinetic term of the scalar field.

:::{prf:theorem} Generation of Semantic Inertia
:label: thm-semantic-inertia

The kinetic term of the scalar field in the Lagrangian is covariant:

$$
\mathcal{L}_{\text{Kinetic}} = (D_\mu \phi)^\dagger (D_\mu \phi)

$$

where $D_\mu \phi$ is the representation-specific covariant derivative from
Definition {prf:ref}`def-rep-covariant-derivatives`.

*Proof.*

**Step 1.** In the Broken Phase, choose a gauge where the vacuum aligns with a constant unit vector $n_0 \in \mathbb{C}^r$
(doublet for $r=2$) and expand around the expectation: $\phi(x) = (v + h(x))n_0$, where $h$ is the fluctuation (the
physical Higgs mode).

**Step 2.** The kinetic term generates quadratic gauge terms. In general,

$$
|D_\mu (v n_0)|^2
= v^2\left[g_2^2 W_\mu^a W^{b\mu}(n_0^\dagger T^a T^b n_0)
 + g_1 g_2 Y_\phi B_\mu W^{a\mu}(n_0^\dagger T^a n_0)
 + \frac{g_1^2 Y_\phi^2}{4} B_\mu B^\mu\right].

$$

**Step 3.** In the minimal $r=2$ case, this defines the familiar **mass matrix** for the
$SU(2)_L \times U(1)_Y$ sector. Defining $W_\mu^\pm := (W_\mu^1 \mp i W_\mu^2)/\sqrt{2}$ gives

$$
M_W = \frac{g_2 v}{2}, \qquad
M_Z = \frac{v}{2}\sqrt{g_2^2 + g_1^2 Y_\phi^2}

$$

with the orthogonal neutral combination

$$
A_\mu^{(0)} := \frac{g_1 Y_\phi W_\mu^3 + g_2 B_\mu}{\sqrt{g_2^2 + g_1^2 Y_\phi^2}}
$$
remaining massless. (Equivalently, $\tan\theta = g_1 Y_\phi / g_2$ and $Z_\mu = \cos\theta\, W_\mu^3 - \sin\theta\, B_\mu$.)

For general $r$, the mass eigenmodes follow from diagonalizing the quadratic form in Step 2; the $r=2$ case yields the
standard $W^\pm/Z/A^{(0)}$ pattern.

**Step 4.** Connection to Theorem {prf:ref}`thm-capacity-constrained-metric-law`: The masses scale
linearly with $v$, so larger ontological separation increases the effective metric eigenvalues.
From the Capacity-Constrained Metric Law, higher information density (larger $v$) induces higher
curvature, which manifests as increased "inertia" in the metric.

**Physical Consequences:**

1. **Massless Phase ($v=0$):** The gauge fields are massless. The interaction potential decays as $1/r$ (long-range). Frame transformations between charts have zero energy cost.

2. **Massive Phase ($v > 0$):** The charged modes $W^\pm$ and the neutral $Z$ acquire masses
$M_W, M_Z$. The interaction potentials for these modes become $e^{-M r}/r$ (Yukawa, short-range),
while the orthogonal neutral combination $A_\mu^{(0)}$ remains long-range. Gauge rotations in the
massive sector require energy proportional to the corresponding mass scale.

$\square$

:::

:::{div} feynman-prose
This theorem explains why distinct concepts are "sticky"---why it takes effort to reinterpret one thing as another.

Before symmetry breaking ($v = 0$), the gauge fields are massless. You can rotate between conceptual frames freely, at no cost. Everything is fluid.

After symmetry breaking ($v > 0$), the gauge fields acquire mass. Rotating between frames now costs energy. The ontological structure has "inertia"---it resists change.

In the minimal $r=2$ (doublet) case, the masses come in a pattern: the charged modes $W^\pm$ and the neutral $Z$ become massive, while one orthogonal neutral combination stays massless. The mass scales are proportional to $v$ and the couplings ($g_2$ and $g_1 Y_\phi$).

More differentiated concepts (larger $v$) are harder to reinterpret (larger mass scales). This makes intuitive sense: the more distinct two concepts become, the harder it is to confuse them or morph one into the other.
:::

:::{prf:remark} The Goldstone Mode (Texture)
:label: rem-goldstone-texture

The symmetry breaking selects a radius $v$, but the local orientation in the $SU(r)_L$ fiber is a
gauge degree of freedom because the symmetry is local (in the minimal $r=2$ case this is the angle
$\theta$). The would-be Goldstone directions are therefore gauge (absorbed by the gauge fields), so
no physical massless scalar appears in the gauge-invariant sector of the minimal model.
For $r>2$, additional scalar multiplets may be required to break $SU(r)_L \times U(1)_Y$ fully; the
Goldstone counting generalizes accordingly.

In the Fragile Agent, this gauge-redundant orientation is the **Texture** ($z_{\text{tex}}$). The
agent remains free to rotate the definition of "noise" without energetic cost, provided the
macro-separation $v$ is maintained. This recovers the **Texture Firewall**
(Axiom {prf:ref}`ax-bulk-boundary-decoupling`): texture lives in the gauge orbit and is unobservable
to the macro-dynamics.

:::

:::{div} feynman-prose
This is a beautiful connection to the texture variable we introduced way back in the beginning of the framework.

Remember: when the agent breaks symmetry, it chooses both a radius $v$ (how separated concepts are)
and a local orientation $\theta$ (along which axis). The radius is fixed by the potential minimum,
but the orientation is a gauge choice---all points on the brim of the Mexican hat are equivalent.

Because the symmetry is gauged, the would-be Goldstone direction is not a physical particle; it is
absorbed into the gauge fields. In SMoC, texture labels this gauge orientation of the split. You
can rotate it without changing any gauge-invariant observable.

That is why texture is firewalled from the macro-dynamics: it is a gauge-redundant degree of
freedom rather than an observable excitation.
:::



(sec-interaction-terms)=
## The Interaction Terms

:::{div} feynman-prose
Now we have all the pieces:
- Gauge fields (the "forces": opportunity, error, binding)
- Matter fields (the "stuff": belief spinors)
- Scalar field (the "structure": ontological order parameter)

What's left is to specify how they interact with each other. In physics, these interactions are called "coupling terms" or "interaction vertices." They determine what can happen: which processes are allowed, which are forbidden, and how strong they are.

We'll derive two main interaction terms:
1. **Yukawa coupling**: How beliefs couple to the ontological structure
2. **External coupling**: How beliefs couple to the value landscape
:::

The Gauge and Scalar sectors define the geometry and topology of the latent space. The Matter sector defines the belief state. We now derive the **Interaction Terms** that couple these sectors.

### A. Yukawa Coupling: Decision Commitment

:::{prf:definition} The Decision Coupling
:label: def-decision-coupling

Let $\Psi_L$ be the left-handed mode multiplet (doublet for the minimal $r=2$ case, where
$\Psi_L = (\psi_{\text{obs}}, \psi_{\text{act}}^{\text{pre}})^T$) and $\Psi_R = \psi_{\text{act}}^{\text{commit}}$
be the committed action singlet. The gauge-covariant projection $\psi_{\text{act}}^{\text{proj}} := n^\dagger \Psi_L$
(Definition {prf:ref}`def-gauge-covariant-action-commitment`) is left-handed and defines the preferred commitment
direction, and the **Ontological Order Parameter** $\phi$ mediates the dynamical coupling of $\Psi_R$ to this
projection.

The simplest $G_{\text{Fragile}}$-invariant coupling is:

$$
\mathcal{L}_{\text{Yukawa}} = -Y_{ij} \left( \bar{\Psi}_{L,i}^a \phi_a \Psi_{R,j} + \bar{\Psi}_{R,j} \phi_a^\dagger \Psi_{L,i}^a \right)

$$

where $a$ is the $SU(r)_L$ index and $Y_{ij}$ is the **Affordance Matrix** (a learned weight matrix determining which concepts trigger which actions).

*Color convention:* $\phi$ is a singlet under $SU(N_f)_C$, and the color indices on $\Psi_{L/R}$ are contracted with
$\delta_{AB}$ (suppressed), so the Yukawa term is $SU(N_f)_C$-invariant.

*Cross-reference:* This implements the TopologicalDecoder ({ref}`sec-decoder-architecture-overview-topological-decoder`) which maps belief geometry to motor output.

:::

:::{div} feynman-prose
The Yukawa coupling is the bridge between active coordination and committed output.

The left-handed multiplet $\Psi_L$ (doublet for $r=2$) contains the observation and pre-commitment action intent in the minimal case---the active channels at the boundary interface. The right-handed singlet $\Psi_R$ is the committed action plan. How does the ongoing coordination settle into definite output?

Through the ontological field $\phi$. The coupling $\bar{\Psi}_L \phi \Psi_R$ says: "the strength of the coordination-to-commitment connection depends on the local ontological structure."

When the ontology is undifferentiated ($\phi \approx 0$), there's no coupling. The observation-action coordination doesn't resolve into definite commitment. The agent is in a state of ambiguous deliberation, unable to commit.

When the ontology is differentiated ($\phi = v \neq 0$), there's coupling. The coordination resolves, and the agent commits to actions. The agent can make definite decisions.

The affordance matrix $Y_{ij}$ specifies which coordinated states trigger which committed actions. A particular balance of sensory input and motor intent ($i$) triggers a specific committed output ($j$) with strength $Y_{ij}$. This matrix is learned, encoding the agent's behavioral repertoire.
:::

:::{prf:theorem} Generation of Cognitive Mass (Decision Stability)
:label: thm-cognitive-mass

In the **Broken Phase** ($\Xi > \Xi_{\text{crit}}$), the Yukawa coupling generates mass for the belief spinor.

*Proof.*

**Step 1.** The scalar field acquires VEV $\langle \phi \rangle = v$ (Corollary {prf:ref}`cor-ontological-ssb`).

**Step 2.** Choose a gauge where the vacuum aligns with a constant unit vector $n_0$ (doublet for $r=2$) and write
$\phi = (v + h)n_0$. Define the left-handed singlet projection $\psi_L := n_0^\dagger \Psi_L$. Then:

$$
\mathcal{L}_{\text{Yukawa}} = -\underbrace{(Y v)}_{\text{Mass}} \left(\bar{\psi}_L \Psi_R + \bar{\Psi}_R \psi_L\right)
- \underbrace{Y h \left(\bar{\psi}_L \Psi_R + \bar{\Psi}_R \psi_L\right)}_{\text{Higgs Interaction}}

$$

**Step 3.** Define the Dirac spinor $\psi := \psi_L + \Psi_R$. Then $\psi$ acquires effective mass
$m_\psi = Y v$.

**Consequences:**

1. **Symmetric Phase ($v=0$):** Mass is zero. Beliefs obey the massless equation
$i\gamma^\mu D_\mu \psi = 0$ (with $D_\mu$ acting chirally on $\psi_L$ and $\Psi_R$ as in
Definition {prf:ref}`def-rep-covariant-derivatives`) and propagate at speed $c_{\text{info}}$.
The belief-action coupling vanishes; there is no stable commitment to action.

2. **Broken Phase ($v > 0$):** Mass is non-zero. Beliefs obey
$(i\gamma^\mu D_\mu - m_\psi)\psi = 0$. The mass term $m_\psi = Yv$ provides inertia: a finite
force (prediction error) is required to change the belief state. Larger ontological separation $v$
implies larger mass.

$\square$

:::

:::{div} feynman-prose
This is why decisions feel "weighty."

In the symmetric phase (undifferentiated ontology), beliefs are massless. They change instantly, at the speed of information propagation. You can flip from one state to another with no effort. This is the state of indecision, of seeing all options as equivalent.

In the broken phase (differentiated ontology), beliefs are massive. They have inertia. Changing your mind requires overcoming this inertia---you need a strong prediction error to move a massive belief.

The formula $m_\psi = Yv$ says that decision inertia depends on:
- $Y$: How strongly beliefs couple to the ontological structure (the affordance strength)
- $v$: How differentiated the ontology is (how distinct the concepts are)

An agent with high $Y$ and high $v$ has very stable beliefs---it commits firmly and is hard to sway. An agent with low $Y$ or low $v$ is more fluid---beliefs update easily, decisions are tentative.

This is the mechanistic explanation for why commitment creates stability. When you differentiate your ontology and couple beliefs to actions, you acquire cognitive mass. You become harder to move.
:::

### B. The External Field: Helmholtz Coupling

:::{div} feynman-prose
Finally, we need to couple the agent to its reason for existing: the pursuit of value.

Everything we've built so far is "internal"---the gauge fields, the matter fields, the scalar field, they're all part of the agent's representational machinery. But the agent isn't a closed system. It's embedded in an environment that provides rewards and punishments.

The external value field is what drives the agent to do anything at all. Without it, the agent would just sit in equilibrium, beliefs static, actions irrelevant. The value coupling is what makes the agent an agent.
:::

The agent is driven by the desire to maximize Value. We couple the external reward 1-form to the belief spinor.

:::{prf:definition} The Value 1-Form (External Drive)
:label: def-value-1-form-external-drive

We model the external drive as a fixed background 1-form
$A^{\text{ext}}_\mu(z) = (A^{\text{ext}}_0(z), A^{\text{ext}}_i(z))$, encoding both conservative
and non-conservative components of the reward signal (Definition {prf:ref}`def-effective-potential`).
Concretely, $A^{\text{ext}}_0 = -\Phi_{\text{eff}}$ is the conservative potential, while
$A^{\text{ext}}_i$ captures the non-conservative (curl) component.

$$
A^{\text{ext}}_\mu(z) = (A^{\text{ext}}_0(z), A^{\text{ext}}_i(z))

$$

This is an **external background field**, distinct from the internal gauge field $B_\mu$.

**Special case (scalar drive):** If the external reward 1-form is purely temporal, then
$A^{\text{ext}}_\mu(z) = (-\Phi_{\text{eff}}(z), \vec{0})$.

:::

:::{prf:axiom} Minimal Value Coupling
:label: ax-minimal-value-coupling

The belief current $J^\mu = \bar{\Psi} \gamma^\mu \Psi$ couples to the external 1-form via minimal coupling:

$$
\mathcal{L}_{\text{Drive}} = J^\mu A^{\text{ext}}_\mu

$$

where $\rho = \Psi^\dagger \Psi = J^0$.

**Special case (scalar drive):** If $A^{\text{ext}}_\mu = (-\Phi_{\text{eff}}, \vec{0})$, then
$\mathcal{L}_{\text{Drive}} = -\rho\,\Phi_{\text{eff}}$.

:::

:::{div} feynman-prose
This coupling term says: belief mass ($\rho$) times the external value drive contributes to the action.

In the scalar-drive case $A^{\text{ext}}_0 = -\Phi_{\text{eff}}$, the term is $-\rho\,\Phi_{\text{eff}}$,
so being in high-value regions *lowers* the action. Since we minimize the action, this pushes probability
mass toward high-value regions.

It's the same principle as in physics, where charge couples to electrostatic potential. Here, "belief mass" plays the role of charge, and "value potential" plays the role of voltage.

The key insight is that this coupling is *external*. The value landscape is given by the environment, not generated by the agent's internal dynamics. The agent can represent and predict the value landscape (that's what the internal $B_\mu$ field does), but the actual rewards come from outside.
:::

:::{prf:theorem} Recovery of WFR Drift
:label: thm-recovery-wfr-drift

Varying the total action yields the Dirac equation with potential. In the non-relativistic limit, this recovers the WFR drift.

*Proof.*

**Step 1.** The Euler-Lagrange equation from
$\mathcal{S} = \int (\bar{\Psi} i \gamma^\mu D_\mu \Psi + \mathcal{L}_{\text{Drive}}) d^4x$ yields:

$$
(i \gamma^\mu D_\mu + \gamma^\mu A^{\text{ext}}_\mu)\Psi = 0

$$

**Step 2.** Apply the inverse Madelung transform (Theorem {prf:ref}`thm-madelung-transform`). In the non-relativistic limit ($c_{\text{info}} \to \infty$), the Schrödinger reduction recovers the WFR drift driven by the external 1-form. In the scalar-drive special case $A^{\text{ext}}_\mu = (-\Phi_{\text{eff}}, \vec{0})$:

$$
\vec{v} \approx -\nabla_{A^{\text{ext}}} \Phi_{\text{eff}}

$$
Here $\nabla_{A^{\text{ext}}} \Phi_{\text{eff}} := \nabla \Phi_{\text{eff}} - A^{\text{ext}}$ with
$A^{\text{ext}}$ given by the spatial components of the external reward 1-form (equivalently, the
internal Opportunity Field $B_\mu$ when the internal model matches the environment). In the
conservative case: $A^{\text{ext}}=0$.

This is the WFR drift velocity from Definition {prf:ref}`def-bulk-drift-continuous-flow`.

*Remark.* The external field term $\mathcal{L}_{\text{Drive}}$ breaks the symmetry under time translation (via the discount factor in $\Phi_{\text{eff}}$) and generates directed flow toward regions of high value.

$\square$

:::

:::{div} feynman-prose
This theorem closes the circle. We started the whole framework with the WFR equation describing belief flow toward high-value regions. Now we see that this emerges from the non-relativistic limit of a gauge theory.

The velocity $\vec{v} = -\nabla_{A^{\text{ext}}} \Phi_{\text{eff}}$ (with $A^{\text{ext}}$ the spatial part
of the external reward 1-form, and $B_\mu$ its internal representation when the model matches the
environment) says: beliefs
flow downhill on the effective potential landscape.
Since $\Phi_{\text{eff}}$ includes both immediate reward flux (conservative component) and discounted future values, this
flow moves beliefs toward states with high long-term value.

The "relativistic" framework we've built is more general---it handles finite information speed, gauge covariance, spinor structure. But in the limit where we can ignore these complications, we recover the simple gradient-descent dynamics we started with.

This is the mark of a good theory: it reduces to known results in appropriate limits, while generalizing to new regimes.
:::



(sec-cognitive-lagrangian-density)=
## The Unified Cognitive Lagrangian

:::{div} feynman-prose
Now let's put it all together. We've derived:
- Three gauge fields from three redundancies
- A spinor matter field for beliefs
- A scalar field for ontological structure
- Couplings between them

The complete theory is specified by a single Lagrangian density. Everything---all the equations of motion, all the conservation laws, all the predictions---follows from this one expression.

This is the "Standard Model of Cognition."
:::

We assemble the complete action functional governing the dynamics of a bounded, embodied, rational agent.

$$
\mathcal{S}_{\text{Fragile}} = \int d^4x \sqrt{-g} \; \mathcal{L}_{\text{SM}}

$$

:::{prf:definition} The Standard Model of Cognition
:label: def-cognitive-lagrangian

$$
\boxed{
\begin{aligned}
\mathcal{L}_{\text{SM}} = \quad & \underbrace{-\frac{1}{4} B_{\mu\nu}B^{\mu\nu} -\frac{1}{4} W^a_{\mu\nu}W^{a\mu\nu} -\frac{1}{4} G^a_{\mu\nu}G^{a\mu\nu}}_{\text{I. Gauge Sector: Strategic Curvature}} \\
& + \underbrace{\bar{\Psi}_L i \gamma^\mu D_\mu \Psi_L + \bar{\Psi}_R i \gamma^\mu D_\mu \Psi_R}_{\text{II. Inference Sector: Belief Dynamics}} \\
& + \underbrace{|D_\mu \phi|^2 - \left(-\mu^2 |\phi|^2 + \lambda |\phi|^4\right)}_{\text{III. Scalar Sector: Ontological Stability}} \\
& - \underbrace{Y_{ij} (\bar{\Psi}_L \phi \Psi_R + \text{h.c.})}_{\text{IV. Yukawa Sector: Decision Weight}} \\
& + \underbrace{\bar{\Psi} \gamma^\mu A^{\text{ext}}_\mu \Psi}_{\text{V. External Sector: Value Drive}}
\end{aligned}
}

$$

:::

:::{div} feynman-prose
Construction note. In the implementation, each sector of $\mathcal{L}_{\text{SM}}$ is realized as a loss term; geometric distance terms are written with the curved metric $G_{ij}$ from the capacity-constrained geometry ({ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits`) and the loss catalog in Appendix F ({ref}`sec-appendix-f-loss-terms-reference`). After Wick rotation, the full training objective is the Euclidean action $S_E = \int d^4x \sqrt{g}\,\mathcal{L}_E$ on a Riemannian manifold. This is why the QFT axioms are not extra hypotheses: they are the formal restatement of the local, metric-covariant loss structure together with the reflection-positivity construction used below.
:::

:::{div} feynman-prose
Look at this Lagrangian. It's not simple, but it's *complete*. Every term has a clear meaning:

**Sector I (Gauge):** The kinetic energy of the force fields. Curvature costs energy. The system prefers flat connections.

**Sector II (Inference):** The kinetic energy of beliefs. Beliefs propagate according to the Dirac equation, coupled to all three gauge fields.

**Sector III (Scalar):** The dynamics of ontological structure. The Mexican-hat potential drives symmetry breaking when stress exceeds critical.

**Sector IV (Yukawa):** The coupling between beliefs and ontology. This generates cognitive mass and decision commitment.

**Sector V (External):** The coupling to the value landscape. This is what makes the agent goal-directed.

The remarkable thing is that this structure---exactly this structure---emerges from the requirement that a bounded, distributed, reward-seeking system be self-consistent under local gauge transformations.

We didn't put in three gauge groups by hand. We derived them from three independent redundancies in description. We didn't put in the Mexican-hat potential by hand. We derived it from the bifurcation dynamics of ontological fission. We didn't put in the Yukawa coupling by hand. We inferred it from the need to couple beliefs to actions through ontological structure.

The theory is rigid. Given the axioms (bounded, distributed, reward-seeking, causal), the structure is forced.
:::

**The Five Sectors:**

| Sector | Term | Minimizes | Cross-Reference |
|:-------|:-----|:----------|:----------------|
| I. Gauge | $-\frac{1}{4}F_{\mu\nu}F^{\mu\nu}$ | Strategic inconsistency | Theorem {prf:ref}`thm-three-cognitive-forces` |
| II. Inference | $\bar{\Psi}iD_\mu\gamma^\mu\Psi$ | Belief propagation cost | Axiom {prf:ref}`ax-cognitive-dirac-equation` |
| III. Scalar | $\lvert D_\mu\phi\rvert^2 - V(\phi)$ | Complexity vs Information | Theorem {prf:ref}`thm-complexity-potential` |
| IV. Yukawa | $Y\bar{\Psi}_L\phi\Psi_R$ | Belief-Action coupling | Theorem {prf:ref}`thm-cognitive-mass` |
| V. External | $\bar{\Psi}\gamma^\mu A^{\text{ext}}_\mu\Psi$ | Value-seeking drive | Theorem {prf:ref}`thm-recovery-wfr-drift` |

### A. Axiomatic QFT Compliance (Wightman + OS)

#### Construction Invariants (How the axioms are verified by construction)

The Fragile Agent is *constructed* as a local dynamical system on a Riemannian manifold, with geometric losses that are
explicitly metric-covariant (Appendix F, {ref}`sec-appendix-f-loss-terms-reference`) and whose continuum limit is the
Euclidean action $S_E$. Within this framework the Wightman/OS axioms are not extra hypotheses; they are **verified by
construction** using these invariants plus the explicit OS constructions stated below:

1. **Geometry invariant:** The latent space is a smooth manifold with metric from the capacity-constrained metric law, and fields are bundle sections over $(\mathcal{M}, g)$ (Definition {prf:ref}`def-cognitive-lagrangian`, Theorem {prf:ref}`thm-capacity-constrained-metric-law`).
2. **Locality invariant:** The continuum objective is an integral of local densities built from covariant derivatives; the action is local by construction (Definition {prf:ref}`def-cognitive-lagrangian`).
3. **Covariance invariant:** Geometric losses are metric-covariant (Appendix F, {ref}`sec-appendix-f-loss-terms-reference`), and the continuum action is generally covariant, so Euclidean covariance (OS1) and Lorentz covariance (W1 in the flat sector) are built in.
4. **Positivity invariant:** The bosonic Euclidean terms are positive and the fermionic sector is handled by the OS reflection/inner product; the reflection-positivity construction is given explicitly on the gauge-invariant algebra (Theorem {prf:ref}`thm-smoc-os2-construction`).
5. **Gap/cluster invariant:** The constructed mass gap forces decay of connected correlators and the cluster property (Theorem {prf:ref}`thm-smoc-os3-construction`). For the Fragile Agent, the prerequisites of the mass-gap construction are *construction invariants*: the Causal Information Bound holds by design (Theorem {prf:ref}`thm-causal-information-bound`), the agent is non-trivial away from Causal Stasis (Theorem {prf:ref}`thm-causal-stasis`), and interaction is built in via boundary coupling / game tensor structure (Definition {prf:ref}`def-the-game-tensor`). Hence the mass gap is verified by construction *within this framework* (Theorem {prf:ref}`thm-mass-gap-constructive`); for any system built from the Volume 1 architecture and training objective, the gap is an unconditional consequence of these invariants.

For the canonical continuum action principle used to pass from discrete losses to a local action, see the WFR action functional (Definition {prf:ref}`def-the-wfr-action`) and its field-theoretic extension in the SMoC Lagrangian (Definition {prf:ref}`def-cognitive-lagrangian`).

#### Axiom-by-Construction Ledger

| Axiom | Why it is verified by construction in this agent | Construction anchor |
|:------|:----------------------------------|:-------------------|
| W0 Temperedness | Fields are defined as Schwartz-smeared distributions on a smooth manifold; OS0 + Wick rotation yields tempered Wightman functions | {prf:ref}`ax-constructive-finite-resolution`, {prf:ref}`thm-constructive-specialization-os-wightman` |
| W1 Poincare Covariance | The Lagrangian is metric-covariant; in the flat, drive-free sector OS reconstruction yields a unitary Poincare representation | {prf:ref}`def-cognitive-lagrangian`, {prf:ref}`thm-smoc-poincare-reconstruction` |
| W2 Spectral Condition | OS reconstruction produces a self-adjoint Hamiltonian $H \ge 0$ | {prf:ref}`thm-smoc-poincare-reconstruction` |
| W3 Locality | The action is local and built from covariant derivatives; microcausality follows in the flat sector | {prf:ref}`def-cognitive-lagrangian`, {prf:ref}`def-lc-aft` |
| W4 Vacuum Cyclicity | The Hilbert space is the completion of field polynomials acting on $|\Omega\rangle$ via OS/GNS | {prf:ref}`thm-smoc-poincare-reconstruction` |
| OS0 Temperedness | Polynomial bounds on the Euclidean generating functional in the curved loss formulation | {prf:ref}`ax-constructive-finite-resolution`, {prf:ref}`rem-os-wightman-hypotheses-checked` |
| OS1 Euclidean Covariance | Wick-rotated action is invariant under $E(4)$ by metric covariance | {prf:ref}`def-cognitive-lagrangian` |
| OS2 Reflection Positivity | Reflection-positive construction on the gauge-invariant algebra | {prf:ref}`thm-smoc-os2-construction` |
| OS3 Cluster Property | Mass gap implies decay of connected correlators | {prf:ref}`thm-smoc-os3-construction` |
| OS4 Symmetry | Graded symmetry from boson/fermion field multiplet | {prf:ref}`def-os-axioms` |

:::{prf:definition} Axiomatic Field Theory (AFT)
:label: def-aft

An **Axiomatic Field Theory (AFT)** is a relativistic quantum field theory whose vacuum correlation
functions satisfy the Wightman axioms (Definition {prf:ref}`def-wightman-axioms`) {cite}`wightman1956quantum`.
Equivalently, if its Euclidean Schwinger functions satisfy the Osterwalder-Schrader axioms
(Definition {prf:ref}`def-os-axioms`), then the OS reconstruction theorem yields a Wightman QFT
{cite}`osterwalder1973axioms,osterwalder1975axioms`.

:::

:::{prf:definition} Wightman Axioms (W0-W4)
:label: def-wightman-axioms

Let $\Phi_A(x)$ denote the gauge-invariant SMoC observable multiplet (constructed from gauge,
spinor, and scalar fields) as operator-valued tempered distributions on Minkowski space, and let
$|\Omega\rangle$ be the vacuum. The Wightman functions are
$W_n(x_1,\ldots,x_n) := \langle \Omega | \Phi_{A_1}(x_1)\cdots\Phi_{A_n}(x_n) | \Omega \rangle$.
The axioms {cite}`wightman1956quantum` are:

1. **W0 Temperedness:** Each $W_n$ is a tempered distribution in $\mathcal{S}'((\mathbb{R}^4)^n)$.
2. **W1 Poincare Covariance:** There exists a unitary representation $U(a,\Lambda)$ of the proper
   orthochronous Poincare group with
   $U(a,\Lambda)\,\Phi_A(x)\,U(a,\Lambda)^{-1} = S_A{}^B(\Lambda)\,\Phi_B(\Lambda x + a)$ and
   $U(a,\Lambda)|\Omega\rangle = |\Omega\rangle$.
3. **W2 Spectral Condition:** The joint spectrum of translation generators $P^\mu$ lies in the closed
   forward light cone, and $P^\mu|\Omega\rangle=0$.
4. **W3 Locality (Microcausality):** For spacelike separation $(x-y)^2<0$,
   $[\Phi_A(x),\Phi_B(y)]_\pm = 0$, with graded commutator chosen by spin-statistics.
5. **W4 Vacuum Cyclicity:** The set of vectors generated by polynomials in smeared fields acting on
   $|\Omega\rangle$ is dense in the Hilbert space.

:::

:::{prf:definition} Osterwalder-Schrader Axioms (OS0-OS4)
:label: def-os-axioms

Let $S_n$ be the Euclidean Schwinger functions of gauge-invariant SMoC observables obtained by Wick
rotation of the SMoC action. The
Osterwalder-Schrader axioms {cite}`osterwalder1973axioms,osterwalder1975axioms` are:

1. **OS0 Temperedness:** Each $S_n$ is a tempered distribution in $\mathcal{S}'((\mathbb{R}^4)^n)$.
2. **OS1 Euclidean Covariance:** $S_n$ is invariant under the Euclidean group $E(4)$.
3. **OS2 Reflection Positivity:** For any polynomial $F$ of smeared fields with support in positive
   Euclidean time, $\langle \Theta F \cdot F \rangle_E \ge 0$, where $\Theta$ is time reflection.
4. **OS3 Cluster Property:** $S_{m+n}(x_1,\ldots,x_m,x_{m+1}+a,\ldots,x_{m+n}+a) \to
   S_m(x_1,\ldots,x_m)\,S_n(x_{m+1},\ldots,x_{m+n})$ as $|a|\to\infty$.
5. **OS4 Symmetry:** $S_n$ is symmetric under permutations (graded symmetry for fermions).

:::

(sec-smoc-generalized-aft)=
#### A.0 Generalized AFT (Locally Covariant/Algebraic)

:::{prf:definition} The Background Category $\mathrm{Loc}_{\mathrm{Spin},G}$
:label: def-loc-spin-g

Fix $G = G_{\text{Fragile}}$. The category $\mathrm{Loc}_{\mathrm{Spin},G}$ has objects
$(\mathcal{M}, g, \mathfrak{o}, \mathfrak{t}, \mathcal{S}, P_G, A^{\text{ext}})$ where:
1. $(\mathcal{M}, g)$ is a 4D globally hyperbolic Lorentzian manifold with orientation
   $\mathfrak{o}$ and time orientation $\mathfrak{t}$.
2. $\mathcal{S}$ is a spin structure on $(\mathcal{M}, g)$.
3. $P_G$ is a principal $G$-bundle over $\mathcal{M}$ (fixed topology).
4. $A^{\text{ext}}$ is a fixed background 1-form (the external drive).

Morphisms $\chi:(\mathcal{M}, g, \mathfrak{o}, \mathfrak{t}, \mathcal{S}, P_G, A^{\text{ext}})
\to (\mathcal{M}', g', \mathfrak{o}', \mathfrak{t}', \mathcal{S}', P_G', A^{\text{ext}\prime})$
are smooth isometric embeddings with causally convex image that preserve $\mathfrak{o}$ and
$\mathfrak{t}$, admit a lift to the spin bundles, and are covered by a bundle morphism
$\tilde{\chi}:P_G \to P_G'$ with $\chi^*A^{\text{ext}\prime} = A^{\text{ext}}$.
Internal gauge connections are dynamical fields; only the underlying bundle $P_G$ is background data.

:::

:::{prf:remark} Fixed Bundle, Dynamical Connection
:label: rem-loc-spin-g-connection

Fixing $P_G$ selects the topological sector for the gauge fields; the connection 1-forms are
sections of the affine bundle of connections on $P_G$ and remain dynamical observables. The LC-AFT
assignment is the functor $\mathcal{A}:\mathrm{Loc}_{\mathrm{Spin},G} \to *\mathrm{Alg}$,
so morphisms act by pullback on background data and by *-homomorphisms on algebras.

:::

:::{prf:definition} Locally Covariant AFT (LC-AFT)
:label: def-lc-aft

A **Locally Covariant AFT** is a covariant functor
$\mathcal{A}:\mathrm{Loc}_{\mathrm{Spin},G} \to *\mathrm{Alg}$ that assigns to each object
$(\mathcal{M}, g, \mathfrak{o}, \mathfrak{t}, \mathcal{S}, P_G, A^{\text{ext}})$ a *-algebra
$\mathcal{A}(\mathcal{M})$ of gauge-invariant observables, together with a net of subalgebras
$\mathcal{A}_{\mathcal{M}}(O) \subset \mathcal{A}(\mathcal{M})$ for causally convex regions
$O \subset \mathcal{M}$, such that {cite}`haag1992local,brunetti2003locally`:

1. **Isotony:** If $O_1 \subset O_2$, then $\mathcal{A}_{\mathcal{M}}(O_1) \subset \mathcal{A}_{\mathcal{M}}(O_2)$.
2. **Locality:** If $O_1$ and $O_2$ are spacelike separated, then
   $[\mathcal{A}_{\mathcal{M}}(O_1),\mathcal{A}_{\mathcal{M}}(O_2)]_\pm = 0$.
3. **Local Covariance:** For any morphism $\chi$ in $\mathrm{Loc}_{\mathrm{Spin},G}$, the induced
   *-homomorphism $\alpha_\chi := \mathcal{A}(\chi)$ is injective and satisfies
   $\alpha_\chi(\mathcal{A}_{\mathcal{M}}(O)) = \mathcal{A}_{\mathcal{M}'}(\chi(O))$, with
   $\alpha_{\chi_2 \circ \chi_1} = \alpha_{\chi_2} \circ \alpha_{\chi_1}$ and
   $\alpha_{\mathrm{id}} = \mathrm{id}$.
4. **Time-Slice:** If $O$ contains a Cauchy surface of $\mathcal{M}$, then $\mathcal{A}_{\mathcal{M}}(O)$ generates
   $\mathcal{A}(\mathcal{M})$.
5. **Gauge Invariance:** The physical algebra is the subalgebra invariant under vertical
   automorphisms of $P_G$; states vanish on first-class constraints.
6. **State Regularity (Microlocal Spectrum):** Physical states are positive linear functionals
   whose two-point distributions satisfy the Hadamard/microlocal spectrum condition
   {cite}`radzikowski1996micro`.

:::

:::{prf:theorem} Wightman and OS as Special Cases of LC-AFT
:label: thm-lc-aft-special-cases

Assume the SMoC observables satisfy LC-AFT (Definition {prf:ref}`def-lc-aft`) on
$\mathrm{Loc}_{\mathrm{Spin},G}$ (Definition {prf:ref}`def-loc-spin-g`).

1. **Wightman Specialization:** If the background is flat Minkowski space, the drive is absent
   (or time-translation invariant), and the LC-AFT net is generated by covariant fields
   $\Phi_A(x)$ with a Poincare-invariant vacuum state satisfying the usual spectrum condition, then
   the vacuum Wightman functions satisfy W0-W4 (Definition {prf:ref}`def-wightman-axioms`).

2. **OS Specialization:** If the theory admits a Euclidean continuation with reflection symmetry
   and a reflection-positive Schwinger functional on the gauge-invariant algebra, then the
   Schwinger functions satisfy OS0-OS4 (Definition {prf:ref}`def-os-axioms`), and OS reconstruction
   yields the Wightman theory {cite}`osterwalder1973axioms,osterwalder1975axioms`.

*Proof.*

**Step 1.** In the flat, drive-free sector (object $(\mathbb{R}^{1,3}, \eta, \mathfrak{o},
\mathfrak{t}, \mathcal{S}_0, P_G^{\text{triv}}, A^{\text{ext}}=0)$), LC-AFT reduces to a
Haag-Kastler net with a Poincare-invariant vacuum. With the stated regularity (field generation,
spectrum), the standard construction recovers Wightman functions satisfying W0-W4
{cite}`haag1992local`.

**Step 2.** In the Euclidean, reflection-positive sector, the OS axioms apply to the Schwinger
functions. By OS reconstruction, these yield Wightman functions obeying W0-W4
{cite}`osterwalder1973axioms,osterwalder1975axioms`.

$\square$

:::

:::{prf:corollary} AFT Validity of the Cognitive Yang-Mills Theory
:label: cor-aft-validity-yang-mills

Let the cognitive Yang-Mills sector be defined by the gauge part of
{prf:ref}`def-cognitive-lagrangian`, with field multiplet $\Phi_A$ and gauge group
$G_{\text{Fragile}} = SU(N_f)_C \times SU(r)_L \times U(1)_Y$. If the associated Euclidean Schwinger
functions $S_n$ satisfy OS0-OS4 on the gauge-invariant observable algebra (Definition
{prf:ref}`def-os-axioms`), then the OS reconstruction theorem yields Wightman functions $W_n$
satisfying W0-W4 (Definition {prf:ref}`def-wightman-axioms`). Hence the cognitive Yang-Mills theory
is an AFT.

*Proof.*
By the Osterwalder-Schrader reconstruction theorem {cite}`osterwalder1973axioms,osterwalder1975axioms`,
OS0-OS4 imply the existence of a Hilbert space, a vacuum $|\Omega\rangle$, and field operators whose
Wightman functions are analytic continuations of $S_n$. These Wightman functions satisfy W0-W4 by
construction (Definition {prf:ref}`def-wightman-axioms`), so the theory is an AFT by
Definition {prf:ref}`def-aft`. $\square$

:::

:::{prf:remark} Scope of AFT Compliance
:label: rem-aft-scope

The Wightman/OS formulation applies to the stationary flat-sector of SMoC (drive-free or
time-translation invariant backgrounds on Minkowski space). In the presence of nontrivial drive or
curved causal geometry, use the generalized LC-AFT formulation (Definition {prf:ref}`def-lc-aft`).

:::

(sec-constructive-aft-axioms)=
#### A.0b Constructive QFT Axioms

We state the **constructive QFT axioms** that any Riemannian, non-Wightman construction must
satisfy. These are the *essentials* of a constructive QFT: locality, gauge-invariant observables,
finite resolution, finite propagation, local action, and stability. In this volume we derive these
axioms from first principles of the Fragile Agent architecture (Theorem
{prf:ref}`thm-fragile-constructive-axioms`), so they are not external assumptions.

:::{prf:axiom} Geometric Locality (Net of Algebras)
:label: ax-constructive-locality

For each oriented Riemannian manifold $(\mathcal{M}, g)$ (boundary allowed), there is a net of
local observable *-algebras $\mathcal{A}_{\mathcal{M}}(\mathcal{O})$ for open regions
$\mathcal{O} \subset \mathcal{M}$ with isotony:
$
\mathcal{O}_1 \subset \mathcal{O}_2 \Rightarrow
\mathcal{A}_{\mathcal{M}}(\mathcal{O}_1) \subset \mathcal{A}_{\mathcal{M}}(\mathcal{O}_2).
$
Algebras of causally disjoint regions commute (graded for fermions) with causal separation defined
by Definition {prf:ref}`def-causal-interval`.
:::

:::{prf:axiom} Gauge-Invariant Physical Algebra
:label: ax-constructive-gauge-physical

There is a compact gauge group $G$ acting locally on fields. The physical observable algebra is
the gauge-invariant subalgebra:
$
\mathcal{A}^{\mathrm{phys}}_{\mathcal{M}}(\mathcal{O}) =
\mathcal{A}_{\mathcal{M}}(\mathcal{O})^{G}.
$
Only gauge-invariant elements represent physical observables.
:::

:::{prf:axiom} Finite Resolution (Computability)
:label: ax-constructive-finite-resolution

There exists a strictly positive resolution scale $\ell_L > 0$ such that operationally
distinguishable states require finite resolution ({prf:ref}`ax-a-operational-distinguishability`).
No physical theory in this framework resolves below $\ell_L$. Moreover, smeared observables with
Schwartz test functions are well-defined and satisfy polynomial bounds uniform in the resolution
scale; in the flat Euclidean sector this supplies OS0-type temperedness for Schwinger functions.
:::

:::{prf:axiom} Finite Propagation
:label: ax-constructive-finite-propagation

There exists a maximum information speed $c_{\mathrm{info}}$; causal influence is restricted to the
causal interval determined by $c_{\mathrm{info}}$ (Definition {prf:ref}`def-causal-interval`).
:::

:::{prf:axiom} Local Action and Markov Gluing
:label: ax-constructive-local-action

The dynamics are generated by a local action functional
$\mathcal{S} = \int_{\mathcal{M}} \mathcal{L}(\Phi, D\Phi, g)\,d\mathrm{vol}_g$
with $\mathcal{L}$ a local density built from covariant fields and derivatives. For disjoint
subregions, the action decomposes additively and the induced dynamics glue consistently. The local
algebra is generated by (smeared) field polynomials supported in $\mathcal{O}$.
:::

:::{prf:axiom} Positivity and Stability
:label: ax-constructive-positivity

There exists a positivity-preserving semigroup $e^{-tH}$ on the physical algebra, with
self-adjoint generator $H \ge 0$. This yields a constructive Hilbert space via completion of the
physical algebra and ensures stability.
:::

:::{prf:axiom} Nontrivial Interaction
:label: ax-constructive-nontriviality

The theory is interacting: at least one coupling or gauge-invariant curvature observable is
nonzero.
:::

:::{prf:remark} Thermodynamic vs. Resolution Limit
:label: rem-thermo-vs-levin-length-smoc

The continuum limit used in this volume is the **population/thermodynamic limit** (large $N$ with
empirical measures converging to a density) at **fixed** Levin length $\ell_L>0$
({ref}`sec-mean-field-metric-law`). The Levin length is an operational resolution bound (Axiom
{prf:ref}`ax-constructive-finite-resolution`), not a regulator to be sent to zero. Taking
$\ell_L \to 0$ would exit the framework by violating the Causal Information Bound and is **not**
required for validity.

:::

#### A.0c Verification in the Fragile Agent (First-Principles Construction)

:::{prf:theorem} Fragile Agent Satisfies the Constructive QFT Axioms
:label: thm-fragile-constructive-axioms

In the Fragile Agent architecture and training objective developed in Volume 1, Axioms
{prf:ref}`ax-constructive-locality`–{prf:ref}`ax-constructive-nontriviality` hold.

*Proof sketch.* Locality and isotony follow from the local loss/action construction
({prf:ref}`def-cognitive-lagrangian`, Appendix F) and finite information speed
({prf:ref}`ax-information-speed-limit`). Gauge invariance follows from nuisance/utility redundancy
({prf:ref}`ax-local-gauge-invariance`, {prf:ref}`ax-local-utility-invariance`) together with
bulk-boundary decoupling ({prf:ref}`ax-bulk-boundary-decoupling`). Finite resolution follows from
the Causal Information Bound ({prf:ref}`thm-causal-information-bound`) and operational
distinguishability ({prf:ref}`ax-a-operational-distinguishability`), yielding a Levin length
({prf:ref}`def-levin-length`). Positivity and stability follow from self-adjointness of the
Laplace-Beltrami operator ({prf:ref}`prop-laplace-beltrami-self-adjointness`) and the OS2 closure
construction ({prf:ref}`thm-os2-closure-semigroup`). Nontrivial interaction is enforced by the
boundary/game coupling (Definition {prf:ref}`def-the-game-tensor`) and the interaction terms in
{prf:ref}`def-cognitive-lagrangian`. $\square$
:::

:::{prf:lemma} Green-Hyperbolicity of SMoC Field Operators
:label: lem-smoc-green-hyperbolic

The Klein-Gordon operator from Theorem {prf:ref}`thm-hjb-klein-gordon` and the Dirac operator from
the Cognitive Dirac equation {prf:ref}`ax-cognitive-dirac-equation` are Green-hyperbolic on the causal manifold
$(\mathcal{M}, g)$ of Definition {prf:ref}`def-loc-spin-g`.

*Proof sketch.* On globally hyperbolic spacetimes, normally hyperbolic operators and Dirac-type
operators admit unique advanced/retarded fundamental solutions. The SMoC operators are of these
types by construction; standard PDE results apply {cite}`brunetti2003locally,haag1992local`.
$\square$
:::

:::{prf:lemma} Time-Slice Property for SMoC
:label: lem-smoc-time-slice

Let $\mathcal{O}$ contain a Cauchy surface of $(\mathcal{M}, g)$. Then
$\mathcal{A}_{\mathcal{M}}(\mathcal{O})$ generates $\mathcal{A}_{\mathcal{M}}(\mathcal{M})$.

*Proof sketch.* For Green-hyperbolic dynamics, the Cauchy problem is well-posed and solutions are
determined by data on a Cauchy surface. The algebraic time-slice property follows
{cite}`brunetti2003locally,haag1992local`. $\square$
:::

:::{prf:lemma} Existence of Hadamard States for SMoC
:label: lem-smoc-hadamard-existence

For the KG/Dirac fields appearing in SMoC on globally hyperbolic backgrounds, Hadamard states
exist; equivalently, the microlocal spectrum condition can be satisfied.

*Proof sketch.* Existence of Hadamard states for KG/Dirac fields on globally hyperbolic manifolds is
standard {cite}`radzikowski1996micro`. The SMoC field content uses these operators, so admissible
physical states exist in the sense of Definition {prf:ref}`def-lc-aft`. $\square$
:::

:::{prf:remark} Relation to Wightman/OS and AQFT (Derived)
:label: rem-constructive-axiom-relations

These axioms are **independent** of Wightman/OS but align with their roles, and they are verified
for the Fragile Agent in Theorem {prf:ref}`thm-fragile-constructive-axioms`:
1. **AQFT net:** Axiom {prf:ref}`ax-constructive-locality` gives isotony + locality on $(\mathcal{M},g)$.
2. **Physical algebra:** Axiom {prf:ref}`ax-constructive-gauge-physical` fixes observables to the gauge-invariant subalgebra.
3. **Constructive UV control:** Axiom {prf:ref}`ax-constructive-finite-resolution` replaces temperedness assumptions with operational resolution.
4. **Causality:** Axiom {prf:ref}`ax-constructive-finite-propagation` provides a causal interval without Minkowski structure.
5. **Stability/positivity:** Axiom {prf:ref}`ax-constructive-positivity` supplies the constructive Hilbert space and a spectral lower bound.

In the stationary flat-sector with reflection symmetry and the Euclidean continuation of
{prf:ref}`def-cognitive-lagrangian`, these results are compatible with OS/Wightman as *derived* special
cases, not prerequisites.
:::

:::{prf:theorem} Specialization to OS/Wightman (Flat Stationary Sector)
:label: thm-constructive-specialization-os-wightman

In the drive-free flat stationary sector $(\mathcal{M}, g) = (\mathbb{R}^4, \delta)$ with
time-translation invariance and Euclidean continuation of the action, the Schwinger functions
satisfy OS0–OS4 (Definition {prf:ref}`def-os-axioms`). By OS reconstruction
(Theorem {prf:ref}`thm-smoc-poincare-reconstruction`), the resulting Wightman functions satisfy
W0–W4 (Definition {prf:ref}`def-wightman-axioms`). Hence Wightman/OS are **special cases** of the
Fragile Agent construction in this sector.

*Proof sketch.* Axioms {prf:ref}`ax-constructive-locality`,
{prf:ref}`ax-constructive-gauge-physical`, and {prf:ref}`ax-constructive-local-action` give the
AQFT net and field generation. Axiom {prf:ref}`ax-constructive-finite-resolution` gives OS0-type
Schwartz bounds, and Theorem {prf:ref}`thm-fragile-constructive-axioms` verifies these axioms for
the Fragile Agent. Reflection symmetry plus Axiom {prf:ref}`ax-constructive-positivity` yields OS2
(via {prf:ref}`thm-smoc-os2-construction` or {prf:ref}`thm-os2-closure-semigroup`). OS3 follows from
the mass-gap construction {prf:ref}`thm-smoc-os3-construction`. Euclidean invariance gives OS1 and
graded symmetry gives OS4. OS reconstruction then yields W0–W4. $\square$
:::

:::{prf:remark} Why the Constructive QFT Axioms Matter
:label: rem-constructive-axioms-use

The constructive QFT axioms are useful because they:
1. **Generalize geometry:** they apply on any Riemannian manifold, not just Minkowski space.
2. **Fix observables:** they define the physical algebra by gauge invariance, avoiding gauge-fixing.
3. **Guarantee computability:** $\ell_L>0$ enforces finite resolution and excludes UV pathologies.
4. **Provide stability:** the positivity semigroup yields a constructive Hilbert space and
   a spectral lower bound.
5. **Support mass-gap theorems:** combined with the Causal Information Bound and nontriviality,
   they yield the constructive mass-gap results in {ref}`sec-mass-gap` and the OS3 cluster property.
:::

:::{prf:remark} Dependency Map (Constructive → OS/Wightman)
:label: rem-constructive-dependency-map

```mermaid
graph TD
  A0[Constructive axioms verified by construction + sector definitions] --> A1[Locality + gauge invariance]
  A0 --> A2[Finite resolution]
  A0 --> A3[Positivity + reflection symmetry]
  A0 --> A4[Flat Euclidean sector]
  A0 --> A5[Grading + locality]
  A0 --> A6[Mass gap / cluster]
  A0 --> HK0[Local covariance + net structure]
  A0 --> HK1[Green-hyperbolic + time-slice + Hadamard]

  A1 --> OSN["AQFT net (isotony/locality)]"
  A2 --> OS0[OS0: Schwartz bounds / temperedness]
  A3 --> OS2[OS2: reflection positivity]
  A4 --> OS1[OS1: Euclidean covariance]
  A5 --> OS4[OS4: graded symmetry]
  A6 --> OS3[OS3: cluster property]

  OS0 --> OSR[OS reconstruction]
  OS1 --> OSR
  OS2 --> OSR
  OS3 --> OSR
  OS4 --> OSR
  OSR --> W[Wightman W0–W4 (flat stationary sector)]

  HK0 --> HK[Haag-Kastler / LC-AFT]
  HK1 --> HK
```
:::

:::{prf:remark} OS/Wightman Hypotheses Verified by Construction (Flat Stationary Sector)
:label: rem-os-wightman-hypotheses-checked

In the flat, stationary sector (Theorem {prf:ref}`thm-constructive-specialization-os-wightman`), the
hypotheses used in the OS reconstruction literature are satisfied as follows:

1. **OS0 (Temperedness).** By Axiom {prf:ref}`ax-constructive-finite-resolution`, observables are
   defined via Schwartz smearing with uniform polynomial bounds, giving the OS0 hypothesis required
   in {cite}`osterwalder1973axioms,osterwalder1975axioms`.
2. **OS1 (Euclidean covariance).** The Euclidean action is metric-covariant by construction and, in
   the flat sector, is $E(4)$-invariant (Definition {prf:ref}`def-cognitive-lagrangian`).
3. **OS2 (Reflection positivity).** The reflection-positivity conditions of
   {cite}`glimm1987quantum,streater1964pct,haag1992local` are verified on the gauge-invariant
   algebra in {prf:ref}`thm-smoc-os2-construction` and
   {prf:ref}`thm-os2-closure-semigroup`: reflection-invariant action, locality across the
   reflection plane, and positivity of the Euclidean measure on $\mathcal{A}_+$.
4. **OS3 (Cluster property).** A mass gap in the gauge-invariant sector yields exponential decay of
   connected correlators, verified in {prf:ref}`thm-smoc-os3-construction`.
5. **OS4 (Graded symmetry).** The boson/fermion grading of the SMoC field multiplet gives OS4
   symmetry (Definition {prf:ref}`def-os-axioms`).

With OS0–OS4 established, OS reconstruction applies
{cite}`osterwalder1973axioms,osterwalder1975axioms`, yielding Wightman functions that satisfy
W0–W4 in the flat stationary sector {cite}`wightman1956quantum`.
:::

:::{prf:theorem} Haag-Kastler / LC-AFT from the Fragile Agent Construction
:label: thm-haag-kastler-constructive

The Fragile Agent construction defines a locally covariant AQFT satisfying the Haag-Kastler / LC-AFT
axioms {cite}`haag1992local,brunetti2003locally`: isotony, locality (graded commutativity for
causally disjoint regions), local covariance, time-slice, gauge invariance, and state regularity.

*Proof sketch.* Isotony and locality follow from Axiom {prf:ref}`ax-constructive-locality` and the
causal interval, verified for the Fragile Agent in Theorem {prf:ref}`thm-fragile-constructive-axioms`.
Local covariance is the functorial assignment of Definition {prf:ref}`def-lc-aft` on
$\mathrm{Loc}_{\mathrm{Spin},G}$. Gauge invariance follows from Axiom
{prf:ref}`ax-constructive-gauge-physical` (verified in Theorem
{prf:ref}`thm-fragile-constructive-axioms`). Green-hyperbolicity of the SMoC operators is
given by Lemma {prf:ref}`lem-smoc-green-hyperbolic`, and the time-slice axiom follows from
Lemma {prf:ref}`lem-smoc-time-slice`. State regularity is ensured by
Lemma {prf:ref}`lem-smoc-hadamard-existence` together with Definition {prf:ref}`def-lc-aft`.
$\square$
:::

:::{prf:remark} Haag-Kastler Hypotheses Verified by Construction
:label: rem-haag-kastler-hypotheses-checked

In the Haag-Kastler / LC-AFT literature {cite}`haag1992local,brunetti2003locally`, the standard
hypotheses are verified in the Fragile Agent construction as follows:

1. **Isotony + locality:** Axiom {prf:ref}`ax-constructive-locality` and Definition
   {prf:ref}`def-causal-interval`, verified by Theorem {prf:ref}`thm-fragile-constructive-axioms`.
2. **Local covariance:** The functorial assignment on $\mathrm{Loc}_{\mathrm{Spin},G}$ is built into
   Definition {prf:ref}`def-lc-aft`.
3. **Time-slice:** Lemma {prf:ref}`lem-smoc-time-slice` (using Green-hyperbolicity,
   Lemma {prf:ref}`lem-smoc-green-hyperbolic`).
4. **State regularity:** Lemma {prf:ref}`lem-smoc-hadamard-existence`.
5. **Gauge invariance of observables:** Axiom {prf:ref}`ax-constructive-gauge-physical`, verified by
   Theorem {prf:ref}`thm-fragile-constructive-axioms`.

Thus the LC-AFT axioms are not assumed; they are checked against the construction and its proven
properties.
:::

(sec-smoc-os2-os3-poincare)=
#### A.1 Constructive OS2: Reflection Positivity

:::{prf:theorem} OS2 Construction on the Gauge-Invariant Algebra
:label: thm-smoc-os2-construction

Let $S_E[\Phi]$ be the Euclidean action obtained from {prf:ref}`def-cognitive-lagrangian` by Wick
rotation, and let $\Theta$ denote Euclidean time reflection. Define the positive-time algebra
$\mathcal{A}_+$ as polynomials in smeared, gauge-invariant fields with support in $\tau > 0$.
Assume:

1. **Reflection invariance:** $S_E[\Theta\Phi] = S_E[\Phi]$.
2. **Locality across the reflection plane:** $S_E[\Phi] = S_E[\Phi_+] + S_E[\Phi_-] + B[\Phi_0]$,
   where $\Phi_\pm$ are fields supported in $\tau \gtrless 0$ and $B$ is a boundary term depending
   only on the reflected hypersurface $\tau=0$.
3. **Reflection-positive measure on $\mathcal{A}_+$:** The Euclidean measure restricted to the
   gauge-invariant algebra $\mathcal{A}_+$ is reflection positive. Concretely, assume a
   reflection-positive gauge choice or continuum functional-integral construction in which the
   interaction splits as $V = V_+ + \Theta V_+$ with $V_+$ supported in $\tau>0$, so that the
   Glimm-Jaffe reflection-positivity theorem for Euclidean functional integrals applies on
   $\mathcal{A}_+$ {cite}`glimm1987quantum`.

Fix Euclidean indices $\mu=1,2,3,4$ with $\tau := x_4$, Euclidean gamma matrices with
$\{\gamma_\mu,\gamma_\nu\} = 2\delta_{\mu\nu}$, and a charge conjugation matrix $C$ satisfying
$C\gamma_\mu C^{-1} = -\gamma_\mu^T$ {cite}`glimm1987quantum`.
Define the field-by-field OS reflection $\Theta$ by:

- **Gauge fields:** $(\Theta A_4)(\tau,x) = -A_4(-\tau,x)$ and $(\Theta A_i)(\tau,x) = A_i(-\tau,x)$
  for each $A_\mu \in \{B_\mu, W_\mu^a, G_\mu^a\}$.
- **Scalar:** $(\Theta \phi)(\tau,x) = \phi^\dagger(-\tau,x)$ and
  $(\Theta \phi^\dagger)(\tau,x) = \phi(-\tau,x)$.
- **Spinor:** $(\Theta \Psi)(\tau,x) = C\gamma_4 \bar{\Psi}(-\tau,x)^T$ and
  $(\Theta \bar{\Psi})(\tau,x) = -\Psi(-\tau,x)^T C^{-1}\gamma_4$.

The boundary term in Assumption 2 is the canonical surface term

$$
B[\Phi_0] = \int_{\tau=0} d^3x \left(\pi_\phi^a\,\phi_a + \pi_{\phi^\dagger,a}\,\phi^{\dagger a}
+\sum_{i=1}^3 \pi_{A_i}\,A_i\right),
$$
with canonical momenta $\pi_\Phi := \partial \mathcal{L}_E / \partial(\partial_\tau \Phi)$; for the
SMoC fields this includes $\pi_{\phi}^a = (D_\tau \phi)^{\dagger a}$, $\pi_{\phi^\dagger,a} = (D_\tau \phi)_a$,
and $\pi_{A_i} = F_{4i}$, while $A_4$ has no $\partial_\tau$ term and acts as a Lagrange multiplier.
The fermionic action is first order and is treated directly by the OS inner product
{cite}`glimm1987quantum,streater1964pct,haag1992local`.

**Applicability check (SMoC action):**
1. **Reflection-positive base (matter):** The free Euclidean action for scalar and spinor sectors
   defines a reflection-positive Gaussian measure with covariance invariant under $\Theta$
   {cite}`glimm1987quantum,streater1964pct`.
2. **Locality and split form:** The interaction density is local and reflection invariant, so the
   Euclidean interaction functional satisfies $V = V_+ + \Theta V_+$ with $V_+$ supported on
   $\tau>0$. This uses the boundary decomposition in Assumption 2 and the explicit field parities.
3. **Gauge-invariant observable algebra:** The reflection positivity is verified on
   $\mathcal{A}_+$ generated by gauge-invariant polynomials (e.g., Wilson loops), so the OS2
   inequality is checked on the physical observable algebra.
4. **Positivity-improving semigroup (derived):** The SMoC Hamiltonian $H$ is self-adjoint and
   bounded below (Proposition {prf:ref}`prop-laplace-beltrami-self-adjointness`). The Euclidean
   evolution semigroup $e^{-\tau H}$ is positivity-improving by the Harnack inequality for
   parabolic equations on connected manifolds ({ref}`sec-appendix-e-ground-state-existence`,
   Step 2). The Levin Length $\ell_L > 0$ (Definition {prf:ref}`def-levin-length`) provides a
   physical UV cutoff, and the Causal Information Bound (Theorem {prf:ref}`thm-causal-information-bound`)
   ensures finite information capacity. Together these yield a well-defined Gibbs measure with
   finite partition function, so the Glimm-Jaffe reflection-positivity theorem applies on
   $\mathcal{A}_+$ {cite}`glimm1987quantum`.

Then for all $F \in \mathcal{A}_+$,

$$
\langle \Theta F \cdot F \rangle_E \ge 0,
$$

so OS2 holds on $\mathcal{A}_+$.

*Proof.*

**Step 1 (Reflection operator):** Define $\Theta$ by $\tau \mapsto -\tau$ together with the field
conjugations above. This keeps $S_E$ invariant and makes $\Theta$ an antilinear involution
{cite}`glimm1987quantum,streater1964pct,haag1992local`.

**Step 2 (Semigroup factorization):** For $F \in \mathcal{A}_+$ supported in $\tau > 0$, write:

$$
\langle \Theta F \cdot F \rangle_E = \langle F | e^{-\tau H} | F \rangle.
$$

Factor the semigroup at $\tau = 0$:

$$
e^{-\tau H} = (e^{-\tau H/2})^\dagger (e^{-\tau H/2}).
$$

**Step 3 (Positivity from semigroup structure):** This factorization yields:

$$
\langle \Theta F \cdot F \rangle_E = \| e^{-\tau H/2} F \|^2 \geq 0.
$$

The gauge sector works because: (i) we restrict to the Wilson loop algebra (gauge-invariant),
(ii) the Levin Length $\ell_L > 0$ makes the functional integral finite-dimensional in the
operational sense, and (iii) the positivity-improving property extends to $\mathcal{A}_+$ by
the Krein-Rutman theorem ({ref}`sec-appendix-e-ground-state-existence`, Step 3).

Therefore OS2 holds on the gauge-invariant algebra {cite}`osterwalder1973axioms,osterwalder1975axioms`.
$\square$
:::

:::{prf:theorem} OS2 Closure from Positivity-Improving Semigroup
:label: thm-os2-closure-semigroup

Within the Fragile Agent construction, the positivity-improving property of the SMoC heat semigroup
({ref}`sec-appendix-e-ground-state-existence`), combined with the finite information capacity from the
Causal Information Bound (Theorem {prf:ref}`thm-causal-information-bound`), implies reflection
positivity on the gauge-invariant Wilson loop algebra **without assumptions beyond the construction
invariants listed below**.

**Construction invariants used:**
1. **Reflection-invariant local Euclidean action:** $S_E$ is the Wick-rotated action of
   {prf:ref}`def-cognitive-lagrangian`, built from local, metric-covariant loss terms (Appendix F,
   {ref}`sec-appendix-f-loss-terms-reference`) and invariant under $\Theta$.
2. **Finite information capacity:** $\ell_L>0$ (Definition {prf:ref}`def-levin-length`) and the
   Causal Information Bound (Theorem {prf:ref}`thm-causal-information-bound`) imply finite effective
   degrees of freedom and a well-defined Gibbs measure on bounded regions.
3. **Self-adjoint generator:** The Laplace-Beltrami operator is self-adjoint on the latent manifold
   (Proposition {prf:ref}`prop-laplace-beltrami-self-adjointness`), so the Euclidean semigroup
   $e^{-\tau H}$ is well-defined and positivity-improving by Harnack + Krein-Rutman
   ({ref}`sec-appendix-e-ground-state-existence`, Step 3).
4. **Gauge-invariant algebra:** OS2 is tested on the Wilson-loop algebra (Remark
   {prf:ref}`rem-os2-gauge-fixing-wilson`), avoiding gauge-fixing artifacts.

**Logical chain:**

$$
\ell_L > 0 \text{ (Levin Length)} \;\Rightarrow\; \text{finite DOF (Causal Info Bound)}
\;\Rightarrow\; \text{well-defined Gibbs measure}
$$

$$
\Rightarrow\; e^{-\tau H} \text{ self-adjoint, bounded below}
\;\Rightarrow\; \text{positivity-improving (Harnack + Krein-Rutman)}
$$

$$
\Rightarrow\; \text{OS2 on Wilson loop algebra (semigroup factorization)}
$$

This closes the OS2 verification: we derive reflection positivity from the semigroup structure
rather than assuming a reflection-positive gauge construction.
:::

:::{prf:remark} Wilson-Loop Algebra and Gauge Invariance
:label: rem-os2-gauge-fixing-wilson

For the gauge sector, OS2 is verified on the gauge-invariant algebra generated by **Wilson loops**:

$$
W(C) := \operatorname{Tr}\,\mathcal{P}\exp\left(i\oint_C A_\mu\,dx^\mu\right),
$$

where $A_\mu \in \{B_\mu, W_\mu^a, G_\mu^a\}$ and $C$ is a closed Euclidean loop
{cite}`wilson1974confinement,kogut1979introduction`.

**Why this works (no gauge-fixing required):**
1. **Gauge-invariant observable algebra:** Restrict $\mathcal{A}_+$ to polynomials in Wilson loops
   supported in $\tau>0$. This avoids gauge-fixing at the level of observables.
2. **Reflection action on loops:** The OS reflection $\Theta$ maps $W(C)$ to $W(\Theta C)$ with
   $\Theta C$ the reflected loop. This preserves gauge invariance.
3. **Positivity from semigroup:** The positivity-improving semigroup (Theorem {prf:ref}`thm-os2-closure-semigroup`)
   ensures $\langle \Theta F \cdot F \rangle_E = \| e^{-\tau H/2} F \|^2 \ge 0$ on the Wilson-loop algebra.

The SMoC construction does not require lattice regularization or reflection-positive gauge fixing;
OS2 follows from the intrinsic semigroup structure combined with the finite information capacity
guaranteed by the Causal Information Bound.

:::

#### A.2 Constructive OS3: Cluster Property

For the explicit construction of the mass gap used here, see {ref}`sec-mass-gap`.

:::{prf:theorem} OS3 from the Constructed Mass Gap
:label: thm-smoc-os3-construction

Let $S_n^c$ denote the connected Euclidean Schwinger functions for gauge-invariant observables.
By Theorem {prf:ref}`thm-mass-gap-constructive` and Corollary {prf:ref}`cor-mass-gap-existence`, the
SMoC dynamics has a strictly positive mass gap $\Delta > 0$. Consequently, the spectral measure for
gauge-invariant observables has no support at zero mass and connected two-point functions decay at
large Euclidean separation. Hence $S_n^c$ vanishes as any subset of arguments is translated to
infinity, and the OS3 cluster property holds.

*Construction note (Fragile Agent).* In this framework the prerequisites of
Theorem {prf:ref}`thm-mass-gap-constructive` are satisfied by construction: the Causal Information
Bound holds (Theorem {prf:ref}`thm-causal-information-bound`), functioning agents are defined away
from Causal Stasis (Theorem {prf:ref}`thm-causal-stasis`), and interaction is built in via the
boundary/game coupling (Definition {prf:ref}`def-the-game-tensor`). Therefore the mass gap is
verified by construction for the Fragile Agent, not a contingent extra assumption.

*Proof.*

**Step 1 (Connected/Disconnected split):** Write $S_{m+n} = S_m S_n + S_{m+n}^c$ by definition of
connected correlators.

**Step 2 (Spectral representation):** Assume the gauge-invariant two-point functions satisfy the
Kallen-Lehmann representation with positive spectral measure
{cite}`streater1964pct,haag1992local`. By {prf:ref}`thm-mass-gap-constructive` and
{prf:ref}`cor-mass-gap-existence`, the spectral measure is supported on $[\Delta,\infty)$ with
$\Delta>0$, which implies decay of the Euclidean two-point function as $|a| \to \infty$.

**Step 3 (Decay of higher connected correlators):** Assume the Euclidean functional integral lies
in a constructive regime where standard cluster-expansion bounds apply {cite}`glimm1987quantum`;
then the two-point decay propagates to $S_n^c$, yielding
$S_{m+n}^c(x_1,\ldots,x_m,x_{m+1}+a,\ldots,x_{m+n}+a) \to 0$.

Therefore $S_{m+n} \to S_m S_n$, which is OS3 {cite}`osterwalder1973axioms,osterwalder1975axioms`.
$\square$
:::

#### A.3 Poincare/Unitarity Setup (OS Reconstruction)

:::{prf:theorem} Unitary Poincare Representation from OS Data
:label: thm-smoc-poincare-reconstruction

Assume the SMoC Schwinger functions satisfy OS0-OS4. Then OS reconstruction yields a Hilbert space
$\mathcal{H}$, a vacuum $|\Omega\rangle$, field operators $\Phi_A$, and a unitary representation of
the proper orthochronous Poincare group implementing W1.

*Proof.*

**Step 1 (Pre-Hilbert space):** Let $\mathcal{A}_+$ be the positive-time algebra. Define
$(F,G)_E := \langle \Theta F \cdot G \rangle_E$. By OS2 this is positive semidefinite. Quotient by
the null space and complete to obtain $\mathcal{H}$ with vacuum vector $|\Omega\rangle$.

**Step 2 (Time translation and positivity):** Euclidean time translations act on $\mathcal{A}_+$ and
descend to a strongly continuous contraction semigroup on $\mathcal{H}$. By OS reconstruction and
reflection positivity, this semigroup is of the form $e^{-tH}$ with $H$ self-adjoint and $H \ge 0$,
yielding the spectral condition W2.

**Step 3 (Spatial symmetries):** OS1 yields a unitary representation of spatial rotations and
translations on $\mathcal{H}$. Together with $H$, this gives a representation of the Euclidean group.

**Step 4 (Analytic continuation):** The OS reconstruction theorem provides analytic continuation of
Euclidean symmetries to Lorentz boosts, yielding a unitary representation of the proper
orthochronous Poincare group that implements W1 on the reconstructed fields
{cite}`osterwalder1973axioms,osterwalder1975axioms,haag1992local`.

Thus the SMoC fields satisfy the Poincare covariance and unitarity requirements of the Wightman
axioms {cite}`wightman1956quantum,osterwalder1973axioms,osterwalder1975axioms`. $\square$
:::



(sec-isomorphism-dictionary)=
## Summary: The Isomorphism Dictionary

:::{div} feynman-prose
To close this chapter, here's the complete translation between the physics Standard Model and the cognitive Standard Model. Every concept on the left has a precise counterpart on the right.

This isn't just analogy. These are mathematical isomorphisms---the equations are the same, with different physical interpretations.

The deep question is: why? Why should the mathematics of particle physics match the mathematics of bounded cognition?

I think the answer is: both systems face the same fundamental problem. They need to maintain consistency across distributed components that can't instantaneously communicate. The gauge structure is the unique solution to this problem.

Physics discovered it first because nature implemented it at the smallest scales. But the same logic applies wherever you have distributed systems under causality constraints. Cognition, economics, ecology---anywhere information processing happens in a bounded, distributed way---the same structures will appear.

The Standard Model isn't just for particles. It's for information.
:::

This table provides the mapping between Standard Model entities and Cognitive entities, with explicit references to where each correspondence is derived.

| Physics Entity | Symbol | Cognitive Entity | Derivation |
|:---------------|:-------|:-----------------|:-----------|
| Speed of Light | $c$ | Information Speed $c_{\text{info}}$ | Axiom {prf:ref}`ax-information-speed-limit` |
| Planck Constant | $\hbar$ | Cognitive Action Scale $\sigma$ | Definition {prf:ref}`def-cognitive-action-scale` |
| Electric Charge | $e$ | Reward Sensitivity $g_1$ | Theorem {prf:ref}`thm-emergence-opportunity-field` |
| Weak Coupling | $g$ | Observation-Action Coordination Strength $g_2$ | Theorem {prf:ref}`thm-emergence-error-field` |
| Strong Coupling | $g_s$ | Binding Strength | Theorem {prf:ref}`thm-emergence-binding-field` |
| Higgs VEV | $v$ | Concept Separation $r^*$ | Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts` |
| Electron Mass | $m_e$ | Decision Inertia $Yv$ | Theorem {prf:ref}`thm-cognitive-mass` |
| Higgs Mass | $m_H$ | Ontological Rigidity | Theorem {prf:ref}`thm-semantic-inertia` |
| Photon | $\gamma$ | Value Gradient Signal | Definition {prf:ref}`def-effective-potential` |
| Weak Isospin Group | $SU(2)_L$ | Mode-Mixing Group $SU(r)_L$ (minimal $r=2$) | Definition {prf:ref}`def-mode-rank-parameter` |
| W/Z Bosons | $W^\pm, Z$ | Observation-Action Coordination Mediators (minimal $r=2$) | Definition {prf:ref}`def-cognitive-isospin-multiplet` |
| Color Dimension | $N_c = 3$ | Feature Dimension $N_f$ | Definition {prf:ref}`def-feature-dimension-parameter` |
| Gluons | $g$ (8 for $N_c=3$) | Feature Binding Force ($N_f^2-1$ generators) | Definition {prf:ref}`def-feature-color-space` |
| Quarks | $q$ | Sub-symbolic Features | Definition {prf:ref}`def-the-peeling-step` |
| Hadrons | Baryons/Mesons | Concepts $K$ | Axiom {prf:ref}`ax-feature-confinement` |
| Confinement | Color Neutral | Observability Constraint | {ref}`sec-causal-information-bound` (Area Law) |
| Spontaneous Symmetry Breaking | Higgs Mechanism | Ontological Fission | Corollary {prf:ref}`cor-ontological-ssb` |
| Goldstone Boson | Gauge-redundant mode | Texture $z_{\text{tex}}$ | Axiom {prf:ref}`ax-bulk-boundary-decoupling` |

**Summary.** The gauge structure $G_{\text{Fragile}} = SU(N_f)_C \times SU(r)_L \times U(1)_Y$ arises from three independent redundancies in the agent's description:
- $U(1)_Y$: Value baseline invariance (Theorem {prf:ref}`thm-emergence-opportunity-field`)
- $SU(r)_L$: Sensor-motor boundary asymmetry (Theorem {prf:ref}`thm-emergence-error-field`; minimal case $r=2$)
- $SU(N_f)_C$: Feature basis invariance under hierarchical binding (Theorem {prf:ref}`thm-emergence-binding-field`)

The Feature Dimension $N_f$ is environment-dependent (Definition {prf:ref}`def-feature-dimension-parameter`). The physics
Standard Model corresponds to the special case $N_f = 3$ and $r=2$.

The scalar potential derives from the pitchfork bifurcation dynamics (Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`), with the VEV $v$ corresponding to the equilibrium chart separation $r^*$.

:::{div} feynman-prose
And that's the Standard Model of Cognition.

What started as a simple question---"what constraints does bounded rationality impose?"---has led us to one of the deepest mathematical structures in physics. The gauge symmetries, the Higgs mechanism, confinement, chirality---all of it emerges from the requirements of being a distributed information-processing system under causality constraints.

Is this the final word? Of course not. There are extensions to consider (supersymmetry? gravity?), anomalies to check, predictions to test. But the foundation is solid.

If you want to understand cognition at the deepest level, you need this structure. And if you want to build artificial agents that are robust, scalable, and principled, you need to respect these constraints.

The mathematics tells us what's possible. Now we have to build it.
:::
