(sec-end-to-end-architecture)=
# End-to-End Architecture: The Universal Geometric Network

## TLDR

**The Synthesis Problem**: This chapter addresses the central engineering challenge: how do we combine gauge-covariant primitives ({ref}`sec-geometric-micro-architecture`) and geodesic dynamics ({ref}`sec-covariant-cross-attention-architecture`) into a complete, end-to-end agent architecture that is both universal (can learn any function) and geometrically consistent (respects the gauge structure $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$)?

**The Universal Geometric Network**: The resolution is a three-stage architecture where strict equivariance lives in the latent dynamics, not at the boundaries. Unconstrained encoders and decoders provide universal approximation by freely choosing and interpreting gauges, while the latent dynamics use *soft equivariance* via L1 regularization on mixing weights. This achieves universal approximation (Theorem {prf:ref}`thm-ugn-universal-approximation`) while maintaining geometric structure through emergent texture zeros—forbidden interactions the network discovers automatically without being told which couplings to suppress.

**Direct Sum Tractability**: For bounded agents, the direct sum representation $\mathcal{Z} = \mathcal{Z}_C \oplus \mathcal{Z}_L \oplus \mathcal{Z}_Y$ with dimension $O(\sum d_i)$ is computationally tractable, whereas the full tensor product $V_C \otimes V_L \otimes V_Y$ with dimension $O(\prod d_i)$ explodes combinatorially. The architecture integrates naturally with the Boris-BAOAB integrator from {ref}`sec-covariant-cross-attention-architecture`, yielding a production-ready system that is both mathematically principled and practically implementable.

## Introduction

:::{div} feynman-prose
Now we face the engineer's question.

We have collected a beautiful toolkit. Spectral linear layers that bound capacity. Norm-gated activations that respect bundle geometry. Isotropic blocks that honor the gauge structure $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$ derived in {ref}`sec-symplectic-multi-agent-field-theory`. We have dynamics—the Lorentz-Langevin equation on the WFR manifold, integrated via the Boris-BAOAB scheme with covariant cross-attention.

But here is the thing: having parts is not the same as having a machine. How do we assemble these pieces into a *complete* agent? Something that takes pixels at one end and produces motor commands at the other. An encoder mapping raw observations into structured latent bundles. A world model predicting how those bundles evolve under actions. A decoder translating internal states back into behavior. The full pipeline.

This is where I want you to see a tension that seems, at first, fatal.

On one hand, we need **universal approximation**. The encoder must be able to represent *any* observation—faces, forests, factory floors, whatever the task demands. The decoder must generate *any* action sequence the task might require. If we constrain the function class too tightly, we cannot learn. Tasks become impossible not because we lack data, but because our architecture literally cannot express the answer.

On the other hand, we need **geometric consistency**. The whole point of our gauge-theoretic framework is that the latent dynamics should respect structure—preserve bundles, maintain capacity bounds, honor the symmetries we derived from first principles. If we allow arbitrary mixing of components, all that beautiful geometry collapses into noise. We would have done all that mathematics for nothing.

Now, these requirements *seem* incompatible. Universal means "can do anything." Geometric consistency means "must obey rules." How can you simultaneously be free and constrained?

Here is the resolution, and it is lovely: **strict equivariance belongs in the latent space, not at the boundaries**. The encoder and decoder are unconstrained—full neural network expressiveness, no geometric restrictions. They are the "interface layers" that translate between the messy external world and the clean internal geometry. But the latent dynamics in the middle? Those respect gauge structure, but *softly*, through L1 regularization that discovers geometric patterns when the task permits them.

Think of it this way. The encoder picks a coordinate system—a "gauge"—for representing the observation. The dynamics evolve the representation according to geometric rules. The decoder translates back into the external world. The encoder and decoder have complete freedom because they are just choosing and interpreting coordinates. The dynamics have structure because physics happens in the middle.

This three-stage architecture—unconstrained boundaries, geometrically structured interior—is the **Universal Geometric Network**. It achieves universal approximation (we will prove this) while maintaining gauge-theoretic consistency (we will prove this too).

And here is the surprise that should make you sit up: the network discovers structure *on its own*. With L1 regularization on the mixing weights, it learns **texture zeros**—forbidden interactions, couplings that get driven to zero—without being told which ones to suppress. The geometry is not a straitjacket forcing a particular structure. It is a *prior* that lets structure emerge when the task calls for it. When the task requires symmetry breaking, the network breaks symmetry. When it does not, the network stays equivariant. The L1 penalty finds the balance automatically.
:::

*Roadmap:* This chapter proceeds in seven sections. First, we define the design space and the fundamental questions any complete architecture must answer (**Section 1**). Then we explore architectural tradeoffs—direct sum versus tensor product representations, levels of cross-bundle interaction, parameter efficiency (**Section 2**). We rigorously establish the limitations of strict equivariance through impossibility theorems (**Section 3**), then present relaxation strategies including approximate equivariance and L1-regularized soft equivariance (**Section 4**). The centerpiece is the Universal Geometric Network itself (**Section 5**), followed by complete implementation details and integration with the BAOAB integrator (**Section 6**). We conclude by connecting to Chapters 04-05 and positioning within the broader literature (**Section 7**).

*Cross-references:* This chapter synthesizes:
- {ref}`sec-geometric-micro-architecture` — Gauge-covariant primitives (SpectralLinear, NormGatedGELU, IsotropicBlock)
- {ref}`sec-covariant-cross-attention-architecture` — Geodesic integrator (BAOAB), covariant cross-attention, Wilson lines
- {ref}`sec-symplectic-multi-agent-field-theory` — Gauge group derivation $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$
- {ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits` — Capacity-constrained metric law
- {ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces` — WFR geometry



(sec-design-space)=
## The Design Space

:::{div} feynman-prose
Let me lay out the design space we are navigating. Think of it as a map with three major forks in the road. At each fork, you have to make a choice, and each choice has consequences—for what the network can learn, what it costs to compute, and whether the geometric structure we care about survives intact.

Some of these decisions are forced by mathematics. If you want gauge invariance, certain structures are mandatory—there is no wriggling out of it. Others involve genuine tradeoffs between expressiveness, computational cost, and geometric fidelity. Understanding which is which will save you from chasing impossible designs.

The central object is the **latent space** $\mathcal{Z}$, where the agent's internal representation lives. Now, this is not just some arbitrary vector space we are free to structure however we like. It has to satisfy constraints:

1. **Bundle decomposition**: $\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i$ where each $V_i \cong \mathbb{R}^{d_b}$ is a feature bundle. Think of these as different "aspects" of the internal state—one bundle might encode position-like information, another velocity-like, another something more abstract.

2. **Gauge symmetry**: Transformations $\rho: G_{\text{Fragile}} \to \text{Aut}(\mathcal{Z})$ that leave physics invariant. You can rotate your coordinate system within each bundle, and nothing observable should change. This is the internal gauge freedom.

3. **Capacity constraint**: Total information $I(X; Z) \leq C$ enforced via spectral bounds. The agent has finite resources—it cannot store arbitrarily precise representations. Spectral normalization keeps signals from blowing up.

4. **Metric structure**: The WFR metric $G_{ij}(z)$ determines how to measure distances and what counts as a straight line (geodesic) in latent space. This is not flat Euclidean space.

With this structure established, here are the three big design questions.

**Question 1: How do gauge factors combine?**

Do we use **direct sum** $\mathcal{Z}_C \oplus \mathcal{Z}_L \oplus \mathcal{Z}_Y$ (independent subspaces sitting side by side) or **tensor product** $V_C \otimes V_L \otimes V_Y$ (entangled quantum numbers, the way particles work in physics)?

The tensor product is "more physical"—it is how quarks carry color, isospin, and hypercharge simultaneously. But it has exponential dimension scaling: if each factor has dimension $d$, the product has dimension $d^3$. Direct sum scales linearly ($3d$). For neural networks with latent dimensions in the hundreds, this matters enormously.

**Question 2: How much symmetry breaking?**

**Strict equivariance** preserves gauge structure exactly. Mathematically beautiful, but as we shall prove, it severely limits what functions the network can compute. **Soft equivariance** encourages but does not enforce symmetry—violations are penalized but allowed. **No equivariance** treats $\mathcal{Z}$ as flat space with no structure, throwing away everything we worked for in the previous chapters.

**Question 3: What level of cross-bundle interaction?**

Even within the direct sum structure, bundles can interact in different ways. **Level 1** (norms only): bundles see only each other's magnitudes—scalar energies, no directional information. **Level 2** (Gram matrix): bundles see relative angles via inner products. **Level 3** (hybrid): mostly equivariant with small learned symmetry-breaking terms.

These questions interact. The answer to Question 1 affects what is possible for Questions 2 and 3. And there is a meta-question hovering over all of them: *where* in the architecture do we enforce constraints? At the boundaries where observations enter and actions exit? In the latent dynamics? Everywhere uniformly?

The answer, as you will see, is: constraints belong in the middle, freedom at the boundaries.
:::

### Formal Definitions

We begin by precisely defining what we mean by a "complete agent architecture."

:::{prf:definition} Complete Agent Architecture
:label: def-complete-agent-architecture

A **complete agent architecture** is a composition of three neural operators:

$$
\mathcal{A}: \mathcal{X} \xrightarrow{E} \mathcal{Z} \xrightarrow{D} \mathcal{Z} \xrightarrow{P} \mathcal{Y}
$$

where:

1. **Encoder** $E: \mathcal{X} \to \mathcal{Z}$ maps observations $x \in \mathcal{X}$ (e.g., images, sensor readings) to latent representations $z \in \mathcal{Z}$

2. **Latent Dynamics** $D: \mathcal{Z} \times \mathcal{Y} \to \mathcal{Z}$ updates latent state given current state and action, implementing one step of the world model or transition function

3. **Policy/Decoder** $P: \mathcal{Z} \to \mathcal{Y}$ maps latent state to outputs $y \in \mathcal{Y}$ (actions, predictions, reconstructions)

**Notation:**
- $\mathcal{X}$ — observation space (typically $\mathbb{R}^{d_x}$ for pixel values)
- $\mathcal{Z}$ — latent space with bundle structure $\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i$
- $\mathcal{Y}$ — output space (actions $\mathcal{U}$ or reconstructed observations)

**Sequential composition:** For multi-step rollouts, the dynamics are applied recursively:

$$
z_0 = E(x_0), \quad z_{t+1} = D(z_t, a_t), \quad a_t = P(z_t)
$$

**Units:**
- $[\mathcal{X}] = $ dimensionless (pixel intensities in $[0, 1]$ or normalized sensor values)
- $[\mathcal{Z}] = \sqrt{\text{nat}}$ (latent space has information-theoretic units; $\sqrt{\text{nat}}$ arises from the Fisher-Rao metric on probability distributions, where distances have units of $\sqrt{\text{information}}$)
- $[\mathcal{Y}] = $ task-dependent (e.g., dimensionless for discrete actions, physical units for continuous control)
:::

:::{prf:definition} Gauge-Equivariant Architecture
:label: def-gauge-equivariant-architecture

An agent architecture $\mathcal{A} = P \circ D \circ E$ is **$G$-equivariant** with respect to gauge group $G$ if:

1. **Latent space transforms:** There exists a representation $\rho: G \to \text{GL}(\mathcal{Z})$ such that for $g \in G$, latent states transform as $z \mapsto \rho(g) z$

2. **Encoder invariance:** $E$ maps to gauge-equivalent latent states:

   $$
   E(x) \sim \rho(g) E(x) \quad \forall g \in G, x \in \mathcal{X}
   $$
   where $\sim$ denotes equivalence up to gauge choice (physically identical states)

3. **Dynamics equivariance:** $D$ commutes with gauge transformations:

   $$
   D(\rho(g) z, a) = \rho(g) D(z, a) \quad \forall g \in G, z \in \mathcal{Z}, a \in \mathcal{Y}
   $$

4. **Decoder covariance:** $P$ produces consistent outputs under gauge transformations:

   $$
   P(\rho(g) z) = P(z) \quad \forall g \in G, z \in \mathcal{Z}
   $$
   (output is gauge-invariant if $\mathcal{Y}$ has no gauge structure)

**Interpretation:** Gauge equivariance ensures that changing the internal coordinate frame (latent representation) doesn't change the agent's observable behavior. Two latent states related by $z' = \rho(g) z$ represent the same physical state in different "mental coordinate systems."

**Example:** For $G = SO(d_b)$ (rotations within bundles), if you rotate all feature vectors by the same matrix $R \in SO(d_b)$, the dynamics and output must transform consistently. The agent's behavior is independent of which orthonormal basis you choose for each bundle.

**Remark:** The encoder invariance condition is weaker than full equivariance because the encoder *chooses* a gauge. Different choices lead to gauge-equivalent latent states. This gauge freedom is crucial for achieving universal approximation ({ref}`sec-universal-geometric-network`).
:::

### The Fundamental Questions

We now frame the design space as a decision tree with three major branches.

| Question | Options | Tradeoff |
|----------|---------|----------|
| **Q1: Bundle composition** | Direct sum $\oplus$ vs Tensor product $\otimes$ | Tractability vs Expressiveness |
| **Q2: Equivariance enforcement** | Strict vs Soft vs None | Symmetry vs Universality |
| **Q3: Cross-bundle interaction** | Norms (Level 1) vs Gram matrix (Level 2) vs Hybrid (Level 3) | Symmetry preservation vs Coupling strength |

:::{prf:definition} Direct Sum Representation
:label: def-direct-sum-representation

The **direct sum** representation structures the latent space as:

$$
\mathcal{Z} = \mathcal{Z}_C \oplus \mathcal{Z}_L \oplus \mathcal{Z}_Y
$$

where:
- $\mathcal{Z}_C \cong \mathbb{R}^{d_C}$ — color subspace (feature bundles)
- $\mathcal{Z}_L \cong \mathbb{R}^{d_L}$ — weak isospin subspace (observation-action doublet)
- $\mathcal{Z}_Y \cong \mathbb{R}^{d_Y}$ — hypercharge subspace (capacity bound)

**Dimension:** $\dim(\mathcal{Z}) = d_C + d_L + d_Y$ (linear scaling)

**Gauge action:** Block-diagonal:

$$
\rho(g_C, g_L, g_Y) = \begin{pmatrix}
\rho_C(g_C) & 0 & 0 \\
0 & \rho_L(g_L) & 0 \\
0 & 0 & \rho_Y(g_Y)
\end{pmatrix}
$$

Each factor acts independently on its subspace.

**Interpretation:** Gauge quantum numbers are independent labels. The color index, isospin index, and hypercharge are separate attributes of a latent state, analogous to storing (position, velocity, temperature) as separate variables.

**Advantage:** Computational tractability—operations scale linearly with total dimension.

**Limitation:** Cannot represent cross-gauge correlations natively. To model "if color is red AND isospin is up, then..." requires learning explicit coupling through network layers.
:::

:::{prf:definition} Tensor Product Representation
:label: def-tensor-product-representation

The **tensor product** representation structures the latent space as:

$$
\mathcal{Z} = V_C \otimes V_L \otimes V_Y
$$

where $V_C, V_L, V_Y$ are the representation spaces for each gauge factor.

**Dimension:** $\dim(\mathcal{Z}) = \dim(V_C) \times \dim(V_L) \times \dim(V_Y)$ (multiplicative scaling)

**Gauge action:** Kronecker product:

$$
\rho(g_C, g_L, g_Y) = \rho_C(g_C) \otimes \rho_L(g_L) \otimes \rho_Y(g_Y)
$$

**Basis:** Each basis vector $|c, \ell, y\rangle$ carries all three quantum numbers simultaneously. Under gauge transformation:

$$
|c, \ell, y\rangle \mapsto \sum_{c', \ell', y'} [\rho_C]_{c'c} [\rho_L]_{\ell'\ell} [\rho_Y]_{y'y} |c', \ell', y'\rangle
$$

**Interpretation:** This is how quantum states work in particle physics. A quark isn't "red" independently of being "up"—it's a single entity transforming under the representation $(\mathbf{3}, \mathbf{2}, 1/6)$ of $SU(3)_C \times SU(2)_L \times U(1)_Y$.

**Advantage:** Can represent entangled states like $\frac{1}{\sqrt{2}}(|r, \uparrow\rangle + |b, \downarrow\rangle)$ where color and isospin are correlated. Full expressiveness of gauge-invariant functions.

**Limitation:** Dimension explodes. For $d_C = 64, d_L = 8, d_Y = 4$: direct sum gives $76$ dimensions, tensor product gives $2048$ dimensions. This is why quarks work with small representations ($3 \times 2 \times 1 = 6$) while neural networks need hundreds of latent dimensions.

**Factorization:** For low-rank structure (when gauge-invariant couplings are sparse), factored tensor representations can recover efficiency:

$$
W = \sum_{k=1}^r U_C^{(k)} \otimes U_L^{(k)} \otimes U_Y^{(k)}
$$
with $r \ll \dim(V_C) \times \dim(V_L) \times \dim(V_Y)$, requiring only $O(r \sum_i \dim(V_i))$ parameters instead of $O(\prod_i \dim(V_i))$.
:::

:::{prf:definition} Cross-Bundle Interaction Levels
:label: def-cross-bundle-interaction-levels

For a latent space $\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i$ with bundles $v_i \in V_i \cong \mathbb{R}^{d_b}$, we define three levels of permissible cross-bundle coupling:

**Level 1: Norms Only** (Strict $\prod_i SO(d_b)_i$ equivariance)

Bundles interact only through their magnitudes $\|v_i\|$. Invariant features:

$$
\mathcal{I}_1 = \{\|v_1\|, \|v_2\|, \ldots, \|v_{n_b}\|\} \subset \mathbb{R}^{n_b}
$$

**Implication:** The output of bundle $i$ has the form (by Schur's lemma, Theorem {prf:ref}`thm-equivariant-function-structure`):

$$
f_i(v_1, \ldots, v_{n_b}) = v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|)
$$
where $\phi_i: \mathbb{R}^{n_b} \to \mathbb{R}$ can be an arbitrary function (e.g., deep MLP).

**Level 2: Gram Matrix** (Global $SO(d_b)$ equivariance)

A single rotation $R \in SO(d_b)$ acts on *all* bundles: $(v_1, \ldots, v_{n_b}) \mapsto (Rv_1, \ldots, Rv_{n_b})$.

Invariant features:

$$
\mathcal{I}_2 = \{G_{ij} = \langle v_i, v_j \rangle : 1 \leq i, j \leq n_b\} \subset \mathbb{R}^{n_b \times n_b}
$$

The Gram matrix $G$ is a symmetric $n_b \times n_b$ matrix with:
- Diagonal: $G_{ii} = \|v_i\|^2$ (energy of bundle $i$)
- Off-diagonal: $G_{ij} = \|v_i\| \|v_j\| \cos\theta_{ij}$ (alignment between bundles $i, j$)

**Implication:** Functions of $G$ can represent any $O(d_b)$-invariant function. Much more expressive than Level 1 (bundles can "see" each other's directions relative to a global frame).

**Level 3: Hybrid / Soft Equivariance**

Use Level 1 (norms) as the primary pathway, add small Level 2 (Gram) or symmetry-breaking terms with regularization:

$$
f = f_{\text{equivariant}} + \lambda \cdot f_{\text{mixing}}
$$

where $\lambda \ll 1$ is learned or regularized (e.g., via L1 penalty).

**Implication:** Network can violate equivariance when necessary for the task, but pays a cost. With strong L1 regularization, symmetry-breaking terms are driven toward zero unless essential.
:::

### Design Space Summary

The following table previews the solution we'll arrive at (the Universal Geometric Network, {ref}`sec-universal-geometric-network`) and contrasts it with the extremes of the design space.

| Design Choice | Strict Equivariant | Universal Geometric Network | Unconstrained Baseline |
|---------------|--------------------|-----------------------------|------------------------|
| **Bundle structure** | Direct sum $\mathcal{Z} = \oplus_i V_i$ | Direct sum $\mathcal{Z} = \oplus_i V_i$ | Flat $\mathcal{Z} = \mathbb{R}^d$ |
| **Encoder equivariance** | Strict | None (encoder chooses gauge) | N/A (no gauge structure) |
| **Latent equivariance** | Strict (norms only) | Soft (L1 regularized mixing) | None |
| **Decoder equivariance** | Invariant outputs | None (decoder interprets gauge) | N/A |
| **Universality** | ✗ (Limited, Theorem {prf:ref}`thm-norm-networks-not-universal`) | ✓ (Proven, Theorem {prf:ref}`thm-ugn-universal-approximation`) | ✓ (Standard UAT) |
| **Geometric consistency** | ✓ (Exact) | ✓ (Soft, with violations $\propto \lambda_{\text{L1}}$) | ✗ |
| **Cross-bundle interaction** | Level 1 (norms) | Level 3 (hybrid: norms + learned mixing) | Arbitrary |
| **Tensor structure** | Direct sum | Direct sum | N/A |
| **Capacity bound** | Spectral normalization | Spectral normalization | None |
| **Emergent structure** | None (fixed by architecture) | L1 discovers texture zeros | None |

**Key insight:** The Universal Geometric Network achieves the best of both worlds by *separating concerns*:
- **Encoder/Decoder** (boundaries): Unconstrained, universal
- **Latent Dynamics** (interior): Softly equivariant, geometrically structured

This decomposition exploits the gauge freedom: the encoder can choose any convenient gauge for representing observations, the latent dynamics respect geometric structure, and the decoder translates back to observable outputs. Gauge transformations in the latent space don't affect encoder or decoder—they're arbitrary internal reframings.



(sec-architectural-choices)=
## Architectural Choices and Tradeoffs

:::{div} feynman-prose
Now I want to walk through the architectural design space systematically, because each choice we make has consequences that ripple through everything else. This is not abstract philosophy—it determines what your network can learn, what it costs to compute, and whether you end up with a machine that respects the geometric structure we have been building.

Let me frame this around the three questions I mentioned.

**The first question is about tensor structure.** We have established that the latent space should decompose into bundles: $\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i$, with each bundle corresponding to a different aspect of the agent's internal state. But here is a subtlety that matters enormously: how do these bundles interact with the gauge symmetry?

In particle physics, a quark is not "red" separately from being "up"—it carries color *and* isospin *and* hypercharge as a single, unified quantum state. The mathematical structure is a tensor product: the state space is $V_C \otimes V_L \otimes V_Y$. But tensor products have a terrible scaling property: dimensions multiply. If $V_C$ has dimension 64, $V_L$ has dimension 8, and $V_Y$ has dimension 4, the tensor product has dimension $64 \times 8 \times 4 = 2048$. Compare that to the direct sum, which gives $64 + 8 + 4 = 76$.

This is not just a factor of 27—it is the difference between "tractable" and "forget it" for neural network architectures. So we face a tradeoff: the tensor product is more faithful to the physics, but the direct sum is actually implementable.

**The second question is about symmetry.** Strict equivariance—where the network exactly respects gauge transformations—has beautiful mathematical properties. But as we will prove shortly, it also has catastrophic limitations on expressiveness. A strictly equivariant network can only compute functions of a very specific form: each output bundle must be the input bundle scaled by some function of all the norms. No rotation within bundles. No direction-dependent cross-talk. Just scalar multiplication.

That sounds limiting because it *is* limiting. So we have two alternatives: soft equivariance (penalize violations but allow them) or no equivariance at all (treat latent space as flat). The answer, as you might guess, is soft equivariance—it gives us the inductive bias of geometry without the straitjacket of strict constraints.

**The third question is about interaction strength.** Even once we commit to direct sum structure with soft equivariance, there is still freedom in how bundles talk to each other. At Level 1 (norms only), bundle $i$ can only see how much energy is in bundles $j$ and $k$—their magnitudes, not their directions. At Level 2 (Gram matrix), bundles can see relative angles through inner products. At Level 3 (hybrid), we add explicit learned mixing with L1 regularization to discover which cross-bundle couplings matter.

Each level is more expressive than the last, and each level breaks more symmetry. The question is: how much expressiveness do you need for your task, and are you willing to pay the geometric cost?

Let me now make these tradeoffs precise.
:::

### Direct Sum vs Tensor Product: Dimensional Scaling

The choice between direct sum and tensor product representations has immediate practical consequences for network capacity and computational cost.

:::{prf:theorem} Dimensional Scaling
:label: thm-dimensional-scaling

For a gauge group $G = G_C \times G_L \times G_Y$ with representation spaces $V_C, V_L, V_Y$ of dimensions $d_C, d_L, d_Y$ respectively:

**Direct sum:**

$$
\dim(\mathcal{Z}_{\oplus}) = d_C + d_L + d_Y = \sum_{i \in \{C, L, Y\}} d_i
$$

**Tensor product:**

$$
\dim(\mathcal{Z}_{\otimes}) = d_C \times d_L \times d_Y = \prod_{i \in \{C, L, Y\}} d_i
$$

*Proof.* By definition of direct sum and tensor product of vector spaces. $\square$
:::

**Concrete comparison:**

| Representation Dimensions | Direct Sum | Tensor Product | Ratio |
|--------------------------|------------|----------------|-------|
| $d_C = 3, d_L = 2, d_Y = 2$ (minimal) | $7$ | $12$ | $1.7\times$ |
| $d_C = 16, d_L = 4, d_Y = 4$ (small) | $24$ | $256$ | $10.7\times$ |
| $d_C = 64, d_L = 8, d_Y = 4$ (medium) | $76$ | $2048$ | $26.9\times$ |
| $d_C = 256, d_L = 16, d_Y = 8$ (large) | $280$ | $32768$ | $117\times$ |

For realistic neural network dimensions, the tensor product becomes prohibitively expensive. This is why particle physics works with small representations (quarks in $(\mathbf{3}, \mathbf{2}, 1/6)$ with dimension $3 \times 2 \times 1 = 6$), while neural networks typically need latent dimensions in the hundreds or thousands.

**Tradeoff table:**

| Aspect | Direct Sum $\oplus$ | Tensor Product $\otimes$ |
|--------|---------------------|--------------------------|
| **Dimension** | $O(\sum d_i)$ linear | $O(\prod d_i)$ exponential |
| **Parameters** | $O((\sum d_i)^2)$ for linear layers | $O((\prod d_i)^2)$ for linear layers |
| **Cross-gauge correlations** | Not natively representable | Natively representable |
| **Gauge action** | Block-diagonal (decoupled) | Kronecker product (entangled) |
| **Interpretation** | Separate quantum numbers | Joint quantum state |
| **Computation** | Tractable for large $d_i$ | Intractable beyond small $d_i$ |
| **Physical fidelity** | Approximation | Exact |

**When tensor product wins:** If the true function has low-rank structure in the tensor product space (e.g., due to gauge-invariant constraints limiting interactions), factored tensor representations can be more parameter-efficient than direct sum learning cross-correlations explicitly.

**Factored Tensor Layer (Low-Rank Alternative):**

```python
class FactoredTensorLayer(nn.Module):
    """Low-rank factorization of tensor product interaction.

    Instead of full W ∈ ℝ^(d_C d_L d_Y) × (d_C d_L d_Y),
    use W = Σ_k U_C^(k) ⊗ U_L^(k) ⊗ U_Y^(k) with rank r << d_C d_L d_Y.

    Parameters: O(r(d_C + d_L + d_Y)) instead of O(d_C d_L d_Y)
    """

    def __init__(self, d_C: int, d_L: int, d_Y: int, rank: int, d_out: int):
        super().__init__()
        self.rank = rank

        # Factored components
        self.U_C = nn.Linear(d_C, rank, bias=False)  # d_C × r
        self.U_L = nn.Linear(d_L, rank, bias=False)  # d_L × r
        self.U_Y = nn.Linear(d_Y, rank, bias=False)  # d_Y × r
        self.U_out = nn.Linear(rank, d_out, bias=False)  # r × d_out

    def forward(self, z_C: Tensor, z_L: Tensor, z_Y: Tensor) -> Tensor:
        """
        Args:
            z_C: [B, d_C] color features
            z_L: [B, d_L] isospin features
            z_Y: [B, d_Y] hypercharge features
        Returns:
            [B, d_out] output
        """
        # Factored trilinear form: Σ_k U_C[z_C]_k · U_L[z_L]_k · U_Y[z_Y]_k · U_out[k]
        interaction = self.U_C(z_C) * self.U_L(z_L) * self.U_Y(z_Y)  # [B, rank], Hadamard product
        return self.U_out(interaction)  # [B, d_out]
```

**Parameter count:** $r(d_C + d_L + d_Y + d_{\text{out}})$ vs $(d_C \times d_L \times d_Y) \times d_{\text{out}}$ for full tensor.

For $d_C = 64, d_L = 8, d_Y = 4, d_{\text{out}} = 64, r = 16$:
- Factored: $16 \times (64 + 8 + 4 + 64) = 2240$ parameters
- Full tensor: $(64 \times 8 \times 4) \times 64 = 131072$ parameters
- **Reduction:** $58.5\times$ fewer parameters

**Recommendation:** For the Universal Geometric Network, use **direct sum** for tractability. Reserve factored tensor layers for specific cross-gauge interactions when empirical evidence suggests low-rank structure.

### Cross-Bundle Interaction Levels

Given a direct sum structure, we still have architectural freedom in how bundles interact.

:::{prf:definition} Level 1: Norms-Only Interaction
:label: def-level1-norms-only

A **norms-only** cross-bundle layer computes outputs solely from bundle magnitudes $\{\|v_i\|\}_{i=1}^{n_b}$.

**Functional form:** For each bundle $i$, the output is:

$$
f_i(v_1, \ldots, v_{n_b}) = v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|)
$$
where $\phi_i: \mathbb{R}^{n_b} \to \mathbb{R}_+$ is an arbitrary positive function (typically a deep MLP with softplus output).

**Equivariance:** Strictly equivariant under $\prod_{i=1}^{n_b} SO(d_b)_i$ (per-bundle rotations).

**Expressiveness:** Can implement energy-based routing (e.g., "suppress bundle 2 when bundles 1 and 3 are both active") but cannot represent direction-dependent cross-talk.
:::

**Implementation:**

```python
class NormInteractionLayer(nn.Module):
    """Level 1: Norms-only cross-bundle interaction."""

    def __init__(self, n_bundles: int, hidden_dim: int = 64):
        super().__init__()
        self.n_bundles = n_bundles

        # MLP: norms → scale factors
        self.norm_mlp = nn.Sequential(
            nn.Linear(n_bundles, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_bundles),
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: [B, n_bundles, bundle_dim]
        Returns:
            [B, n_bundles, bundle_dim]
        """
        norms = torch.norm(z, dim=-1)  # [B, n_bundles]
        scales = F.softplus(self.norm_mlp(norms))  # [B, n_bundles], positive
        return z * scales.unsqueeze(-1)  # [B, n_bundles, bundle_dim]
```

:::{prf:definition} Level 2: Gram Matrix Interaction
:label: def-level2-gram-matrix

A **Gram matrix** interaction layer uses the full matrix of inner products $G_{ij} = \langle v_i, v_j \rangle$.

**Invariant features:**

$$
\mathcal{I}_2 = \{G_{ij} : 1 \leq i, j \leq n_b\} \subset \mathbb{R}^{n_b \times n_b}
$$

**Equivariance:** Equivariant under global $SO(d_b)$ acting on all bundles simultaneously: $(v_1, \ldots, v_{n_b}) \mapsto (Rv_1, \ldots, Rv_{n_b})$ for single $R \in SO(d_b)$.

**Not equivariant** under per-bundle rotations $(R_1 v_1, \ldots, R_{n_b} v_{n_b})$ with independent $R_i$.

**Expressiveness:** Can encode relative orientations between bundles. Much more expressive than norms-only.
:::

**Implementation:**

```python
class GramInteractionLayer(nn.Module):
    """Level 2: Gram matrix cross-bundle interaction."""

    def __init__(self, n_bundles: int, hidden_dim: int = 64):
        super().__init__()
        self.n_bundles = n_bundles
        gram_dim = n_bundles * n_bundles

        self.mlp = nn.Sequential(
            nn.Linear(gram_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_bundles),
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: [B, n_bundles, bundle_dim]
        Returns:
            [B, n_bundles, bundle_dim]
        """
        # Compute Gram matrix G_{ij} = ⟨v_i, v_j⟩
        gram = torch.bmm(z, z.transpose(-1, -2))  # [B, n_b, n_b]
        gram_flat = gram.view(z.shape[0], -1)  # [B, n_b²]

        # MLP predicts scale factors
        scales = F.softplus(self.mlp(gram_flat))  # [B, n_b]
        return z * scales.unsqueeze(-1)
```

:::{prf:definition} Level 3: Hybrid / Soft Equivariance
:label: def-level3-hybrid

A **hybrid** interaction layer combines:
1. An equivariant pathway (norms-only, Level 1)
2. A mixing pathway (Gram-based or fully learned) with L1 regularization

**Functional form:**

$$
f(z) = f_{\text{equiv}}(z) + f_{\text{mix}}(z)
$$

where $f_{\text{equiv}}$ is strictly equivariant and $f_{\text{mix}}$ has learned weights $W$ penalized by $\lambda_{\text{L1}} \|W\|_1$.

**Equivariance:** Soft—violations are controlled by L1 penalty strength.

**Expressiveness:** Universal (in limit of $\lambda_{\text{L1}} \to 0$), but biased toward equivariant solutions.
:::

This is exactly the `SoftEquivariantLayer` implemented in the UGN (Section {ref}`sec-universal-geometric-network`).

**Comparison table:**

| Level | Invariants | Equivariance Group | Expressiveness | Computational Cost |
|-------|------------|--------------------|--------------------|-------------------|
| **1: Norms** | $\{\|v_i\|\}$ | $\prod_i SO(d_b)_i$ (per-bundle) | Limited | Cheap: $O(n_b d_b + h^2)$ |
| **2: Gram** | $\{G_{ij}\}$ | $SO(d_b)$ (global, single rotation) | High | Moderate: $O(n_b^2 d_b + h^2)$ |
| **3: Hybrid** | $\{\|v_i\|\}$ + learned $W$ | Soft $\prod_i SO(d_b)_i$ (L1 regularized) | Universal | Moderate: $O(n_b^2 d_b^2)$ |

where $h$ is MLP hidden dimension.

**Recommendation:** Use Level 3 (hybrid/soft) as default. The equivariant pathway provides inductive bias, the mixing pathway provides flexibility, and L1 lets the network discover the right balance.



(sec-strict-equivariance-limitations)=
## Strict Equivariance: Beauty and Limitations

:::{div} feynman-prose
This is where beauty meets reality, and reality wins. At least if you care about being able to learn arbitrary functions.

Strict equivariance is mathematically gorgeous. I want you to appreciate that before I demolish it. Every layer respects the gauge structure exactly. No violations, no approximations, perfect geometric consistency. The output always transforms correctly when you rotate the input. And there is a deep theorem from representation theory—Schur's lemma—that tells you exactly what functions you are allowed to compute under these constraints.

The problem is that "exactly what functions you are allowed to compute" turns out to be a *very small* class.

Here is the brutal fact. If you insist on strict per-bundle $SO(d_b)$ equivariance—meaning each bundle can be rotated independently and the network must behave consistently—then your network can only compute functions of a very specific form:

$$f_i(\text{all bundles}) = v_i \cdot \phi_i(\|v_1\|, \|v_2\|, \ldots, \|v_{n_b}\|)$$

Each output bundle is the corresponding input bundle, scaled by some function of all the norms. That is *it*. No mixing of components between bundles. No rotation within bundles. No direction-dependent cross-talk. The network sees only scalar magnitudes—how much energy is in each bundle—and multiplies each bundle by a learned scale factor that depends on those magnitudes.

Stop and think about what this means. Your network cannot learn "if bundle 1 points north, do X; if it points east, do Y." It cannot tell north from east! It can only tell "large" from "small." It cannot learn to rotate the camera, because internal rotations are invisible to the network. It cannot learn that certain directions of motion lead to reward, because all directions look the same.

This seems like a catastrophic limitation for building anything useful. And it is. But it is not a bug in the theory—it is a deep fact about symmetry. Schur's lemma is not something you can argue with. If you want equivariance, this is what you get.

So we face a choice: accept the limitations, or relax the constraints. As I will show you, relaxation via L1 regularization is the way out.

But first, let me prove these claims rigorously. I want you to see *why* strict equivariance is so limiting, not just take my word for it.
:::

### Mathematical Framework: Schur's Lemma for Bundles

:::{prf:theorem} Schur's Lemma for Bundle Representations
:label: thm-schur-bundle

Let $V = \bigoplus_{i=1}^{n_b} V_i$ where each $V_i \cong \mathbb{R}^{d_b}$ is an irreducible representation of $SO(d_b)_i$ (the $i$-th copy of $SO(d_b)$ acting only on $V_i$).

Let $T: V \to V$ be a linear map that is equivariant with respect to $\prod_{i=1}^{n_b} SO(d_b)_i$:

$$
\rho(g) \circ T = T \circ \rho(g) \quad \forall g \in \prod_{i=1}^{n_b} SO(d_b)_i
$$
where $\rho(g)(v_1, \ldots, v_{n_b}) = (g_1 v_1, \ldots, g_{n_b} v_{n_b})$ is the diagonal representation of $g = (g_1, \ldots, g_{n_b})$.

Then $T$ must be **block-diagonal** with each block $T_i: V_i \to V_i$ satisfying $T_i = \lambda_i I_{d_b}$ for some scalar $\lambda_i \in \mathbb{R}$.

*Proof.*

**Step 1 (Block-diagonal structure):** By the direct sum decomposition, $T$ must respect the bundle structure:

$$
T = \begin{pmatrix}
T_{11} & T_{12} & \cdots & T_{1n_b} \\
T_{21} & T_{22} & \cdots & T_{2n_b} \\
\vdots & \vdots & \ddots & \vdots \\
T_{n_b 1} & T_{n_b 2} & \cdots & T_{n_b n_b}
\end{pmatrix}
$$
where $T_{ij}: V_j \to V_i$.

**Step 2 (Off-diagonal blocks vanish):** Consider $g = (I, \ldots, I, g_j, I, \ldots, I)$ where only the $j$-th component is non-identity, and let $v = (0, \ldots, 0, v_j, 0, \ldots, 0)$ with only the $j$-th bundle nonzero.

By linearity, $T(v) = (T_{1j}(v_j), T_{2j}(v_j), \ldots, T_{n_b,j}(v_j))$.

Equivariance requires:

$$
T(g \cdot v) = g \cdot T(v)
$$

The LHS is:

$$
T(0, \ldots, 0, g_j v_j, 0, \ldots, 0) = (T_{1j}(g_j v_j), \ldots, T_{n_b,j}(g_j v_j))
$$

The RHS is:

$$
g \cdot T(v) = (g_1 T_{1j}(v_j), \ldots, g_i T_{ij}(v_j), \ldots, g_{n_b} T_{n_b,j}(v_j))
$$

For $i \neq j$, equating components gives:

$$
T_{ij}(g_j v_j) = g_i T_{ij}(v_j) \quad \forall g_i \in SO(d_b), g_j \in SO(d_b), v_j \in V_j
$$

**Key observation:** For any fixed $g_j$ and $v_j$, the LHS $T_{ij}(g_j v_j)$ is a **fixed vector** in $V_i$. But the RHS $g_i T_{ij}(v_j)$ can be **any rotation** of $T_{ij}(v_j)$ as we vary $g_i$ arbitrarily over $SO(d_b)$. For these to be equal for all choices of $g_i$, we need:

$$
T_{ij}(g_j v_j) = g_i T_{ij}(v_j) \quad \text{for all } g_i \in SO(d_b)
$$

This means $T_{ij}(v_j)$ must be **invariant** under all rotations $g_i$ (i.e., $T_{ij}(v_j)$ lies in the fixed-point set of $SO(d_b)$ acting on $V_i$). For $d_b \geq 2$, the only vector invariant under all rotations is the zero vector. Therefore $T_{ij}(v_j) = 0$ for all $v_j$.

Therefore, $T_{ij} = 0$ for all $i \neq j$.

**Step 3 (Diagonal blocks are scalar multiples):** For each $i$, the block $T_{ii}: V_i \to V_i$ must commute with all $g_i \in SO(d_b)_i$:

$$
T_{ii}(g_i v_i) = g_i T_{ii}(v_i) \quad \forall g_i \in SO(d_b)_i, v_i \in V_i
$$

By Schur's lemma for irreducible representations, $SO(d_b)$ acting on $\mathbb{R}^{d_b}$ is irreducible over $\mathbb{R}$ (for $d_b \geq 2$), so any intertwining operator must be a scalar multiple of the identity:

$$
T_{ii} = \lambda_i I_{d_b}
$$

**Conclusion:**

$$
T = \begin{pmatrix}
\lambda_1 I_{d_b} & 0 & \cdots & 0 \\
0 & \lambda_2 I_{d_b} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_{n_b} I_{d_b}
\end{pmatrix}
$$

$\square$

**Corollary:** Any equivariant linear layer can only scale each bundle independently. No within-bundle mixing, no cross-bundle mixing at the linear level.
:::

### The Universal Approximation Problem

:::{prf:theorem} Norm-Based Networks Are NOT Universal
:label: thm-norm-networks-not-universal

A feedforward network with:
- Input: $z = (v_1, \ldots, v_{n_b}) \in \bigoplus_{i=1}^{n_b} \mathbb{R}^{d_b}$
- Layers: Each strictly equivariant under $\prod_i SO(d_b)_i$
- Activations: Applied per-bundle (e.g., norm-gating)

can only approximate functions of the form:

$$
f_i(v_1, \ldots, v_{n_b}) = v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|)
$$
where $\phi_i: \mathbb{R}^{n_b} \to \mathbb{R}$ is an arbitrary continuous function.

Such networks are **not universal approximators** over continuous functions $f: \mathbb{R}^{n_b \cdot d_b} \to \mathbb{R}^{n_b \cdot d_b}$.

*Proof.*

**Step 1 (Induction setup):** Consider a depth-$L$ network. We prove by induction that each layer output has the form $f_i = v_i \cdot \phi_i(\text{norms})$.

**Base case ($L = 1$):** By Theorem {prf:ref}`thm-schur-bundle`, the first layer can only scale bundles: $z^{(1)}_i = \lambda_i v_i$. If $\lambda_i$ depends on norms $\{\|v_j\|\}$ (via a norm-MLP), we get $z^{(1)}_i = v_i \cdot \phi_i^{(1)}(\|v_1\|, \ldots, \|v_{n_b}\|)$.

**Inductive step:** Suppose layer $\ell$ outputs $z^{(\ell)}_i = v_i \cdot \phi_i^{(\ell)}(\|v_1\|, \ldots, \|v_{n_b}\|)$. Layer $\ell+1$ can only scale:

$$
z^{(\ell+1)}_i = z^{(\ell)}_i \cdot \psi_i^{(\ell+1)}(\|z^{(\ell)}_1\|, \ldots, \|z^{(\ell)}_{n_b}\|)
$$

But $\|z^{(\ell)}_j\| = \|v_j\| \cdot |\phi_j^{(\ell)}(\|v_1\|, \ldots, \|v_{n_b}\|)|$, which is itself a function of norms only. Thus:

$$
z^{(\ell+1)}_i = v_i \cdot \underbrace{\phi_i^{(\ell)}(\{\|v_j\|\}) \cdot \psi_i^{(\ell+1)}(\{\|v_j\| \cdot |\phi_j^{(\ell)}(\{\|v_k\|\})|\})}_{\phi_i^{(\ell+1)}(\{\|v_j\|\})}
$$

The composition is still a function of norms only.

**Step 2 (Counterexample):** Consider the target function:

$$
f(v_1, v_2) = v_{1,1} \cdot v_{2,1}
$$
where $v_{1,1}, v_{2,1}$ denote the first components of bundles 1 and 2.

This function depends on **specific components**, not just norms. For example:
- $v_1 = (1, 0), v_2 = (1, 0)$: $f = 1$
- $v_1 = (1, 0), v_2 = (0, 1)$: $f = 0$

Both inputs have $\|v_1\| = \|v_2\| = 1$, but produce different outputs. Any function $g(\|v_1\|, \|v_2\|)$ must give the same value for both, so $f$ cannot be approximated by norm-based networks.

**Step 3 (Non-density conclusion):** The class of functions representable by norm-based networks is **not dense** in $C(\mathbb{R}^{n_b d_b}, \mathbb{R}^{n_b d_b})$ with respect to the topology of uniform convergence on compact sets. Intuitively, norm-based functions are parametrized by mappings $\mathbb{R}^{n_b} \to \mathbb{R}^{n_b}$ (the norm-to-scale functions $\phi_i$), while general continuous functions on $\mathbb{R}^{n_b d_b}$ form an infinite-dimensional space with vastly more degrees of freedom. The counterexample above (which cannot be approximated) demonstrates a gap in the closure, proving non-density and hence non-universality.

$\square$
:::

:::{div} feynman-prose
Let me make sure you understand what this theorem is saying, because it is devastating for anyone who hoped strict equivariance was the answer.

The counterexample tells the whole story. Take two inputs: $v_1 = (1, 0)$ and $v_2 = (1, 0)$ versus $v_1 = (1, 0)$ and $v_2 = (0, 1)$. Both have the same norms: $\|v_1\| = 1$ and $\|v_2\| = 1$. But the function $f = v_{1,1} \cdot v_{2,1}$ (the product of the first components) gives different answers: 1 in the first case, 0 in the second.

A strictly equivariant network cannot tell these inputs apart. It sees only magnitudes, not directions. So it must give the same output for both. But the target function does not. Therefore, the target function cannot be learned.

This is not a failure of our particular architecture. It is a mathematical *impossibility*. No matter how deep you make the network, no matter how wide, no matter what nonlinearities you use—if every layer is strictly equivariant under per-bundle rotations, the network is fundamentally blind to directions within bundles. And "blind to directions" means "cannot learn direction-dependent functions."

The theorem quantifies this precisely: the class of learnable functions has *measure zero* in the space of all continuous functions. You are not missing a few corner cases—you are missing almost everything.

This is why we need soft equivariance.
:::

:::{prf:theorem} Equivariant Function Structure
:label: thm-equivariant-function-structure

For a function $f: \bigoplus_{i=1}^{n_b} \mathbb{R}^{d_b} \to \bigoplus_{i=1}^{n_b} \mathbb{R}^{d_b}$ to be equivariant under $\prod_{i=1}^{n_b} SO(d_b)_i$, it must satisfy:

$$
f_i(R_1 v_1, \ldots, R_{n_b} v_{n_b}) = R_i f_i(v_1, \ldots, v_{n_b}) \quad \forall R_j \in SO(d_b), v_j \in \mathbb{R}^{d_b}
$$

The **necessary and sufficient** form is:

$$
f_i(v_1, \ldots, v_{n_b}) = v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|)
$$
where $\phi_i: \mathbb{R}^{n_b} \to \mathbb{R}$ is arbitrary.

*Proof.*

**Sufficiency:** Compute:

$$
f_i(R_1 v_1, \ldots, R_i v_i, \ldots, R_{n_b} v_{n_b}) = R_i v_i \cdot \phi_i(\|R_1 v_1\|, \ldots, \|R_{n_b} v_{n_b}\|)
$$

Since $\|R_j v_j\| = \|v_j\|$ (orthogonal matrices preserve norms):

$$
= R_i v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|) = R_i \left[ v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|) \right] = R_i f_i(v_1, \ldots, v_{n_b})
$$

**Necessity:** Fix all bundles except $i$: $v_j = c_j$ for $j \neq i$. Equivariance under $R_i$ gives:

$$
f_i(c_1, \ldots, c_{i-1}, R_i v_i, c_{i+1}, \ldots, c_{n_b}) = R_i f_i(c_1, \ldots, c_{i-1}, v_i, c_{i+1}, \ldots, c_{n_b})
$$

This must hold for all $R_i \in SO(d_b)$ and all $v_i$. By Schur's lemma (irreducibility of $SO(d_b)$ on $\mathbb{R}^{d_b}$), $f_i$ must be proportional to $v_i$:

$$
f_i(\ldots, v_i, \ldots) = v_i \cdot \psi_i(\ldots, v_i, \ldots)
$$

But $\psi_i$ must also be equivariant under $R_i$:

$$
\psi_i(\ldots, R_i v_i, \ldots) = \psi_i(\ldots, v_i, \ldots)
$$

This is only possible if $\psi_i$ depends on $v_i$ through $\|v_i\|$ alone (the only $SO(d_b)$-invariant feature). Extending to all bundles:

$$
f_i = v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|)
$$

$\square$
:::

### What CAN Norm-Based Networks Represent?

Despite the limitations, norm-based networks are still expressive within their function class.

**Capabilities:**

| Task | Representable? | Example |
|------|----------------|---------|
| Energy-based routing | ✓ | $f_i = v_i$ if $\|v_i\| > \theta$, else $0$ |
| Competition | ✓ | $f_i = v_i \cdot \mathbb{1}[\|v_i\| = \max_j \|v_j\|]$ |
| Cooperation | ✓ | $f_i = v_i \cdot \sigma(\sum_j \|v_j\| - \theta)$ |
| Nonlinear norm interactions | ✓ | $\phi_i(\|\mathbf{v}\|)$ can be arbitrarily complex MLP |
| **Rotation within bundle** | ✗ | Cannot learn "rotate $v_1$ by 45°" |
| **Direction-dependent cross-talk** | ✗ | Cannot implement $f_1 = v_1$ if $v_2$ points north, else $0$ |
| **Component mixing** | ✗ | Cannot set $f_{1,1} = v_{2,3}$ (first component of output bundle 1 = third component of input bundle 2) |

**Theorem (Capabilities):**

:::{prf:proposition} Expressiveness of Norm-Based Networks
:label: prop-norm-network-capabilities

A norm-based equivariant network with $L$ layers and hidden dimension $h$ can approximate any continuous function $\Phi: \mathbb{R}^{n_b} \to \mathbb{R}^{n_b}$ (the norm-to-scale mapping) to arbitrary precision, by the universal approximation theorem for MLPs.

Thus, norm-based networks are **universal** over the restricted class:

$$
\mathcal{F}_{\text{norm}} = \left\{ f: f_i = v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|), \; \phi_i \in C(\mathbb{R}^{n_b}, \mathbb{R}) \right\}
$$

But $\mathcal{F}_{\text{norm}}$ has measure zero in $C(\mathbb{R}^{n_b d_b}, \mathbb{R}^{n_b d_b})$.
:::



(sec-relaxation-strategies)=
## Relaxation Strategies

:::{div} feynman-prose
Here is the elegant resolution to the expressiveness limitation.

We have just proved that strict equivariance is too limiting—your network can only compute functions of a very specific form, and that form cannot represent most of what you want to learn. But wait: abandoning geometric structure entirely throws away all the beautiful properties we worked so hard to derive. The gauge consistency, the capacity bounds, the connection to physics—all gone if we just treat latent space as flat $\mathbb{R}^n$.

What we need is a middle path. Not strict equivariance (too limiting) and not no equivariance (throws away structure). We need **approximate** or **soft** equivariance.

The idea is simple but powerful. Instead of enforcing $f(Rz) = Rf(z)$ exactly for all rotations $R$—an ironclad constraint—we allow small violations. The network can break symmetry when the task requires it, but we penalize the magnitude of symmetry-breaking through the loss function. This gives us a tunable tradeoff: crank up the penalty and you get near-exact equivariance; reduce it and you get more flexibility.

Think of it like a spring attached to the equivariant solution. The equivariant structure is the equilibrium position—natural, low-energy, geometrically clean. Pulling away from that equilibrium costs you (through regularization loss). But sometimes the task reward you gain by breaking symmetry is worth the cost. The network finds the optimal balance automatically during training.

The beautiful thing is that L1 regularization—the same tool used for sparse feature selection—turns out to discover *which* symmetry-breaking terms matter. Most get driven to zero. A few survive because the task needs them. And the pattern of zeros and nonzeros that emerges tells you something about the underlying structure of your problem.

This is the key to the Universal Geometric Network. Not rigid constraints, but soft pressure toward geometric structure that yields when necessary.

Let me show you the options for implementing this.
:::

### Approximate Equivariance

:::{prf:definition} Approximate Equivariance
:label: def-approximate-equivariance

A function $f: \mathcal{Z} \to \mathcal{Z}$ is **$\epsilon$-approximately equivariant** with respect to group $G$ and representation $\rho$ if:

$$
\sup_{g \in G, z \in \mathcal{Z}} \frac{\|f(\rho(g) z) - \rho(g) f(z)\|}{\|z\|} \leq \epsilon
$$

The quantity:

$$
\mathcal{V}(f) := \mathbb{E}_{g \sim G, z \sim \mathcal{Z}} \left[ \|f(\rho(g) z) - \rho(g) f(z)\|^2 \right]
$$
is the **equivariance violation**.

**Remark:** The supremum measure gives a worst-case bound, while $\mathcal{V}(f)$ measures average-case violation used in optimization.
:::

:::{prf:theorem} Approximate Equivariance Bound
:label: thm-approximate-equivariance-bound

Let $f_{\text{equiv}}: \mathcal{Z} \to \mathcal{Z}$ be strictly $G$-equivariant and $f_{\text{break}}: \mathcal{Z} \to \mathcal{Z}$ be an arbitrary symmetry-breaking term.

Define:

$$
f = f_{\text{equiv}} + \lambda f_{\text{break}}
$$

Then the equivariance violation satisfies:

$$
\mathcal{V}(f) = \lambda^2 \mathbb{E}_{g \sim \mu_G, z} \left[ \|f_{\text{break}}(\rho(g) z) - \rho(g) f_{\text{break}}(z)\|^2 \right]
$$
where $\mu_G$ is the Haar measure on $G$ (uniform distribution for compact Lie groups)

*Proof.*

**Step 1:** Compute the violation:

$$
f(\rho(g) z) - \rho(g) f(z) = f_{\text{equiv}}(\rho(g) z) + \lambda f_{\text{break}}(\rho(g) z) - \rho(g) [f_{\text{equiv}}(z) + \lambda f_{\text{break}}(z)]
$$

**Step 2:** Use equivariance of $f_{\text{equiv}}$:

$$
= \rho(g) f_{\text{equiv}}(z) + \lambda f_{\text{break}}(\rho(g) z) - \rho(g) f_{\text{equiv}}(z) - \lambda \rho(g) f_{\text{break}}(z)
$$

$$
= \lambda [f_{\text{break}}(\rho(g) z) - \rho(g) f_{\text{break}}(z)]
$$

**Step 3:** Square and take expectation:

$$
\mathbb{E}_{g, z} \|f(\rho(g) z) - \rho(g) f(z)\|^2 = \lambda^2 \mathbb{E}_{g, z} \|f_{\text{break}}(\rho(g) z) - \rho(g) f_{\text{break}}(z)\|^2
$$

$\square$

**Implication:** The violation scales quadratically with $\lambda$. Small symmetry-breaking terms ($\lambda \ll 1$) lead to small violations.
:::

**Four practical options:**

1. **Accept violations:** Train without explicit equivariance penalty. Monitor $\mathcal{V}(f)$ as a diagnostic. If task is mostly equivariant, violations should be small at convergence.

2. **Soft loss penalty:** Add $\mathcal{L}_{\text{equiv}} = \lambda_{\text{equiv}} \mathcal{V}(f)$ to training loss. Network trades off task performance vs symmetry preservation.

3. **Strict equivariance for critical layers:** Use strict equivariant layers (norms-only) for latent dynamics, allow violations in encoder/decoder. This is the UGN strategy.

4. **Hybrid architecture:** Decompose $f = f_{\text{equiv}} + f_{\text{mix}}$ explicitly, apply L1 penalty to mixing pathway. This is soft equivariance via regularization.

**Comparison:**

| Strategy | Pros | Cons | Use Case |
|----------|------|------|----------|
| **Accept** | Simple, no hyperparameters | No explicit geometric prior | Exploratory phase |
| **Soft penalty** | Tunable tradeoff | Requires choosing $\lambda_{\text{equiv}}$ | When violation tolerance is known |
| **Strict latent** | Guaranteed equivariance in dynamics | Encoder/decoder unconstrained | UGN default |
| **Hybrid + L1** | Emergent structure discovery | Requires L1 schedule tuning | When structure is unknown |

### L1 Regularization for Emergent Structure

The most sophisticated relaxation strategy uses L1 regularization to let the network **discover** the right amount of symmetry breaking.

:::{prf:theorem} L1 and Hierarchies
:label: thm-l1-hierarchies

Consider a network with mixing weights $W \in \mathbb{R}^{n_b \times n_b \times d_b \times d_b}$ (cross-bundle coupling) trained with loss:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{L1}} \|W\|_1
$$

Under mild regularity conditions (task loss differentiable, optimizer converges), the following hold at convergence:

1. **Sparsity:** Under L1 regularization, most weights are driven near zero. The expected number of weights with $|W_{ij}^{(k\ell)}| \geq \epsilon$ scales inversely with $\lambda_{\text{L1}}$ (heuristically: stronger regularization → fewer large weights), though the exact bound depends on task gradient structure.

2. **Texture zeros:** Cross-bundle blocks $(i, j)$ where task loss $\mathcal{L}_{\text{task}}$ is insensitive to $W_{ij}$ have $\|W_{ij}\|_F \approx 0$ (driven to zero by L1 with no opposing gradient).

3. **Hierarchy:** Non-zero weights organize into levels with exponential decay (approximately): if $|W^{(1)}| > |W^{(2)}| > \cdots$ are sorted magnitudes, then $|W^{(k+1)}| / |W^{(k)}| \approx \text{const} < 1$.

*Informal proof sketch:*

**(1) Sparsity:** L1 penalty creates a "soft thresholding" effect. Weights with $|W| < \lambda_{\text{L1}} / |\partial \mathcal{L}_{\text{task}} / \partial W|$ are driven to zero (penalty gradient dominates task gradient). As $\lambda_{\text{L1}}$ increases, fewer weights exceed this threshold, though the exact scaling depends on the task loss landscape.

**(2) Texture zeros:** If $\partial \mathcal{L}_{\text{task}} / \partial W_{ij} = 0$ (block irrelevant to task), the total gradient is purely from L1: $\partial \mathcal{L}_{\text{total}} / \partial W_{ij} = \lambda_{\text{L1}} \cdot \text{sign}(W_{ij})$, which pushes $W_{ij} \to 0$.

**(3) Hierarchy:** During training, once a weight crosses zero (becomes active), it can grow under task gradients. But nearby small weights continue to be suppressed by L1. This creates a "rich get richer" dynamic: large weights grow, small weights vanish. The distribution becomes hierarchical.

**Empirical validation required.** This theorem is heuristic; rigorous proof requires analyzing stochastic gradient dynamics.
:::

**Implementation: L1 scheduling**

```python
class L1Scheduler:
    """Adaptive L1 regularization schedule.

    If violation > target: increase λ to suppress mixing (reduce violations)
    If violation < target: decrease λ to allow more mixing (increase expressiveness)
    """

    def __init__(
        self,
        lambda_init: float = 0.01,
        target_violation: float = 0.1,
        adaptation_rate: float = 0.01,
    ):
        self.lambda_L1 = lambda_init
        self.target = target_violation
        self.alpha = adaptation_rate

    def step(self, current_violation: float):
        """Update λ_L1 based on measured violation.

        If violation > target: increase λ to suppress mixing
        If violation < target: decrease λ to allow more mixing
        """
        error = current_violation - self.target
        self.lambda_L1 *= (1 + self.alpha * error)  # Multiplicative update
        self.lambda_L1 = max(1e-6, min(1.0, self.lambda_L1))  # Clamp
        return self.lambda_L1
```

**Training protocol:**

1. **Warmup (epochs 1-10):** Start with low $\lambda_{\text{L1}} = 0.001$. Let network explore, establish basic coupling structure.

2. **Ramp up (epochs 10-50):** Gradually increase $\lambda_{\text{L1}}$ to encourage sparsity. Monitor task loss—if it plateaus, stop increasing.

3. **Adaptive (epochs 50+):** Use L1Scheduler to maintain target violation $\mathcal{V} \approx 0.1$ (or other chosen tolerance).

4. **Fine-tune (final epochs):** Fix $\lambda_{\text{L1}}$, train with early stopping on validation loss.

**Diagnostic outputs (every N steps):**

```python
def log_sparsity_diagnostics(model, step):
    """Log mixing weight statistics."""
    for i, layer in enumerate(model.latent_layers):
        W_mix = layer.mixing_weights  # [n_b, n_b, d_b, d_b]

        # Sparsity ratio (fraction of near-zero weights)
        epsilon = 1e-3
        total = W_mix.numel()
        near_zero = (W_mix.abs() < epsilon).sum().item()
        sparsity = near_zero / total

        # Texture zeros (blocks with Frobenius norm < ε)
        n_b = W_mix.shape[0]
        texture_zeros = []
        for bi in range(n_b):
            for bj in range(n_b):
                if bi != bj:
                    block_norm = torch.norm(W_mix[bi, bj], p='fro').item()
                    if block_norm < epsilon:
                        texture_zeros.append((bi, bj))

        # Hierarchy (magnitude distribution)
        magnitudes = W_mix.abs().flatten().sort(descending=True)[0]
        top_10 = magnitudes[:10].tolist()

        log_dict = {
            f'layer_{i}/sparsity': sparsity,
            f'layer_{i}/texture_zeros': len(texture_zeros),
            f'layer_{i}/top_10_magnitudes': top_10,
        }
        wandb.log(log_dict, step=step)
```



(sec-universal-geometric-network)=
## The Universal Geometric Network

:::{div} feynman-prose
After all that exploration of the design space—direct sum versus tensor product, strict versus soft equivariance, norms versus Gram matrices—here is the synthesis. This is the architecture that makes everything work together.

The **Universal Geometric Network** has three stages, and the three stages have different jobs.

**Stage 1: The Encoder.** Raw observations come in—pixels, sensor readings, whatever the world presents. The encoder maps these to latent space. And here is the crucial point: the encoder is *unconstrained*. It is just a neural network (with spectral normalization for capacity bounds). It can represent any continuous function.

What the encoder does, in the language of gauge theory, is **choose a gauge**. It picks an internal coordinate system for representing the observation. Different observations might use different gauges. That is fine—gauge choice is arbitrary, as long as you are consistent about how you handle it downstream.

**Stage 2: Latent Dynamics.** This is where the geometry lives. The latent layers respect gauge structure via soft equivariance: each layer combines an equivariant pathway (norm-based, strictly respecting bundle rotations) plus a mixing pathway (learned cross-bundle couplings with L1 regularization).

The L1 penalty is doing something subtle and important. It encourages most mixing weights to be zero—only the couplings that actually help the task survive. The network discovers which cross-bundle interactions matter and suppresses the rest. This gives us **texture zeros** (forbidden couplings) and **hierarchical structure** (a few strong couplings, many weak ones) without hard-coding any of it.

**Stage 3: The Decoder.** Latent states get mapped to outputs—actions, predictions, whatever the task requires. The decoder is also *unconstrained*. It can represent any function from latent space to output space.

The decoder's job, in gauge terms, is to **interpret the gauge**. It extracts observable quantities from the latent representation. Different latent states related by gauge transformations (internal reframings) should produce the same output, because gauge is a choice of description, not a physical difference.

Now, why does this three-stage structure solve our problem?

Here is the insight: **gauge transformations in latent space are internal**. They do not affect what the encoder receives as input, and they do not affect what the decoder produces as output. The encoder picks a gauge. The dynamics respect that gauge (softly, with L1-discovered violations). The decoder reads off observables in whatever gauge was established.

This is exactly how gauge theories work in physics. You pick a gauge (say, Lorenz gauge or Coulomb gauge for electromagnetism), write gauge-covariant equations for the dynamics, and compute gauge-invariant observables at the end. The choice of gauge is a convenience, not a physical fact.

The Universal Geometric Network implements this principle in neural architecture. Boundaries are free, the middle is geometric. And the combination achieves what seemed impossible: universal approximation *and* geometric consistency.
:::

### The Key Insight

:::{admonition} Design Principle: Gauge Freedom and Universal Approximation
:class: tip
:name: design-gauge-freedom-universality

For an architecture $\mathcal{A} = P \circ D \circ E$ where:
- $E: \mathcal{X} \to \mathcal{Z}$ is an unconstrained encoder
- $D: \mathcal{Z} \to \mathcal{Z}$ is a $G$-equivariant latent dynamics
- $P: \mathcal{Z} \to \mathcal{Y}$ is an unconstrained decoder

the composition $\mathcal{A}$ can achieve universal approximation even when $D$ has restricted function class, exploiting:

1. **Gauge freedom**: For any $z \in \mathcal{Z}$ and $g \in G$, states $z$ and $\rho(g)z$ are physically equivalent

2. **Encoder gauge choice**: $E$ can map $x \in \mathcal{X}$ to any gauge-equivalent representative $z \in [z]_G$ (orbit under $G$)

3. **Decoder gauge invariance**: $P(\rho(g)z) = P(z)$ for all $g \in G, z \in \mathcal{Z}$

**Intuition:** To approximate a target function $f: \mathcal{X} \to \mathcal{Y}$:
- Encoder learns to map $x \mapsto z$ in a gauge that makes the target function "simple" for the restricted latent dynamics $D$
- Latent dynamics implements (possibly multi-step) processing within the gauge-chosen latent space
- Decoder learns to extract $y = f(x)$ from the latent representation

The bottleneck is not expressiveness (encoder/decoder handle that) but *efficient encoding*—whether the encoder can find a gauge where the target function is representable by $D$. For "natural" tasks with geometric structure, such gauges exist. For arbitrary functions, the encoder and decoder absorb all complexity.

**Example:** Suppose $D$ can only rotate and scale bundles. To approximate a complex nonlinear function $f$, the encoder maps inputs to a latent space where $f$ is approximately "rotation + scaling," and the decoder undoes this preprocessing. The encoder/decoder pair effectively linearizes the problem for $D$.

**Formal justification:** See {prf:ref}`thm-ugn-universal-approximation` for rigorous proof.
:::

This design principle suggests our architecture: use unconstrained networks for $E$ and $P$, and soft-equivariant layers for $D$. We now formalize this.

### Formal Specification

:::{prf:definition} Universal Geometric Network
:label: def-universal-geometric-network

The **Universal Geometric Network** (UGN) is a three-stage architecture:

$$
\mathcal{A}_{\text{UGN}}: \mathcal{X} \xrightarrow{E} \mathcal{Z} \xrightarrow{D_1, \ldots, D_L} \mathcal{Z} \xrightarrow{P} \mathcal{Y}
$$

**Stage 1: Encoder** (Unconstrained)

$$
E: \mathbb{R}^{d_x} \to \mathbb{R}^{n_b \cdot d_b}
$$
Implemented as:

$$
E(x) = \text{SpectralMLP}(x) = W_2 \sigma(W_1 x)
$$
where $W_1, W_2$ have spectral norm $\|W_i\|_2 \leq 1$ and $\sigma = \text{GELU}$ (smooth, non-polynomial).

**Output structure:** $z = E(x) \in \mathcal{Z}$ with implicit bundle decomposition $z = (v_1, \ldots, v_{n_b})$ where $v_i \in \mathbb{R}^{d_b}$.

**Stage 2: Latent Dynamics** (Soft Equivariant)

Each layer $D_\ell: \mathcal{Z} \to \mathcal{Z}$ has the form:

$$
D_\ell(z) = D_\ell^{\text{equiv}}(z) + D_\ell^{\text{mix}}(z)
$$

where:
- **Equivariant pathway** $D_\ell^{\text{equiv}}$: Strict $\prod_i SO(d_b)$-equivariant (norm-based, Definition {prf:ref}`def-cross-bundle-interaction-levels` Level 1)
- **Mixing pathway** $D_\ell^{\text{mix}}$: Weakly equivariant or symmetry-breaking (Gram-based or learned)

**Regularization:** L1 penalty on mixing pathway weights:

$$
\mathcal{L}_{\text{reg}} = \lambda_{\text{L1}} \sum_{\ell=1}^L \|W_\ell^{\text{mix}}\|_1
$$

**Stage 3: Decoder** (Unconstrained)

$$
P: \mathbb{R}^{n_b \cdot d_b} \to \mathbb{R}^{d_y}
$$
Implemented as:

$$
P(z) = \text{SpectralMLP}(z) = W_4 \sigma(W_3 z)
$$
with spectral normalization $\|W_i\|_2 \leq 1$.

**Total loss:**

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}}(y, \hat{y}) + \lambda_{\text{L1}} \mathcal{L}_{\text{reg}} + \lambda_{\text{equiv}} \mathcal{L}_{\text{equiv}}
$$

where:
- $\mathcal{L}_{\text{task}}$ — task loss (e.g., MSE for regression, cross-entropy for classification)
- $\mathcal{L}_{\text{reg}} = \sum_\ell \|W_\ell^{\text{mix}}\|_1$ — L1 regularization on mixing weights
- $\mathcal{L}_{\text{equiv}} = \mathbb{E}_{z, R} \|D(Rz) - RD(z)\|^2$ — equivariance violation penalty (optional)

**Hyperparameters:**
- $n_b$ — number of bundles (typically 4-8)
- $d_b$ — bundle dimension (typically 8-64)
- $L$ — number of latent layers (typically 3-6)
- $\lambda_{\text{L1}}$ — L1 regularization strength (typically 0.001-0.1)
- $\lambda_{\text{equiv}}$ — equivariance penalty (typically 0 or 0.01)

**Implementation note:** When $\lambda_{\text{equiv}} = 0$, equivariance is encouraged only through L1 (which drives mixing weights to zero, making layers effectively equivariant). When $\lambda_{\text{equiv}} > 0$, we explicitly penalize equivariance violations, providing a stronger geometric prior.
:::

### Simplified Implementation Example

:::{note}
:class: feynman-added
**Implementation Note:** This section provides a **simplified implementation** where all bundles have the same dimension (`bundle_dim`). For a **production implementation with heterogeneous bundle dimensions**, see Section 6 which uses `List[BundleConfig]` to support bundles of different sizes.
:::

We provide production-ready PyTorch code for the Universal Geometric Network.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class UGNConfig:
    """Configuration for Universal Geometric Network.

    Args:
        input_dim: Input dimension (e.g., flattened image pixels)
        output_dim: Output dimension (e.g., number of actions)
        n_bundles: Number of feature bundles [dimensionless]
        bundle_dim: Dimension of each bundle [dimensionless]
        n_latent_layers: Number of soft-equivariant latent layers
        lambda_l1: L1 regularization strength on mixing weights
        lambda_equiv: Equivariance violation penalty (0 = no explicit penalty)
    """
    input_dim: int
    output_dim: int
    n_bundles: int = 4
    bundle_dim: int = 16
    n_latent_layers: int = 4
    lambda_l1: float = 0.01
    lambda_equiv: float = 0.0


class SpectralLinear(nn.Module):
    """Linear layer with spectral normalization (σ_max ≤ 1).

    Preserves U(1)_Y hypercharge via Lipschitz constraint.
    See Definition 4.3 in Section 04 for full specification.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Apply spectral normalization
        self.linear = nn.utils.spectral_norm(self.linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SoftEquivariantLayer(nn.Module):
    """Latent layer with soft equivariance via L1 regularization.

    Architecture:
        Output = equivariant_path(z) + mixing_path(z)

    where equivariant_path is strictly SO(d_b)-equivariant (norms only)
    and mixing_path breaks symmetry with L1 penalty.

    Args:
        n_bundles: Number of bundles [dimensionless]
        bundle_dim: Dimension per bundle [dimensionless]
        hidden_dim: Hidden dimension for norm MLP
    """

    def __init__(
        self,
        n_bundles: int,
        bundle_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_bundles = n_bundles
        self.bundle_dim = bundle_dim

        # === Equivariant pathway (norms → scales) ===
        self.norm_mlp = nn.Sequential(
            nn.Linear(n_bundles, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_bundles),
        )

        # === Mixing pathway (L1 regularized) ===
        # Block mixing matrix: maps bundle j to contribution to bundle i
        # Shape: [n_bundles, n_bundles, bundle_dim, bundle_dim]
        # W[i, j] maps v_j -> component added to v_i
        self.mixing_weights = nn.Parameter(
            torch.randn(n_bundles, n_bundles, bundle_dim, bundle_dim) * 0.01
        )

        # === Activation ===
        self.gate_bias = nn.Parameter(torch.zeros(n_bundles))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, n_bundles, bundle_dim]
        Returns:
            [B, n_bundles, bundle_dim]
        """
        B, n_b, d_b = z.shape
        device = z.device

        # === Equivariant pathway ===
        norms = torch.norm(z, dim=-1)  # [B, n_b]
        scales = F.softplus(self.norm_mlp(norms))  # [B, n_b]
        z_equiv = z * scales.unsqueeze(-1)

        # === Mixing pathway ===
        # Zero out diagonal (self-mixing handled by equivariant path)
        mask = (1 - torch.eye(n_b, device=device)).view(n_b, n_b, 1, 1)
        mixing = self.mixing_weights * mask

        # Apply mixing: z_mix[i] = Σ_j W[i,j] @ z[j]
        z_mix = torch.einsum('ijkl,bjl->bik', mixing, z)

        # === Combine ===
        z_out = z_equiv + z_mix

        # === Norm-gated activation ===
        norms_out = torch.norm(z_out, dim=-1, keepdim=True) + 1e-8  # [B, n_b, 1]
        gates = F.gelu(norms_out.squeeze(-1) + self.gate_bias)  # [B, n_b]

        # Normalize then re-scale by gate
        z_out = z_out / norms_out  # Unit vectors
        z_out = z_out * gates.unsqueeze(-1)  # Gated magnitude

        return z_out

    def l1_loss(self) -> torch.Tensor:
        """L1 penalty on off-diagonal mixing weights.

        Uses Frobenius norm per block: ||W[i,j]||_F for i ≠ j
        This is group lasso: encourages entire blocks to be zero.
        """
        n_b = self.n_bundles
        loss = torch.tensor(0.0, device=self.mixing_weights.device)

        for i in range(n_b):
            for j in range(n_b):
                if i != j:
                    # Frobenius norm of the [bundle_dim × bundle_dim] block
                    loss = loss + torch.norm(self.mixing_weights[i, j], p='fro')

        return loss

    def mixing_strength(self) -> float:
        """Total strength of mixing pathway.

        Returns:
            Scalar in [0, ∞) measuring symmetry breaking.
            0 = fully equivariant, >0 = symmetry violation.
        """
        n_b = self.n_bundles
        total = 0.0

        for i in range(n_b):
            for j in range(n_b):
                if i != j:
                    total += torch.norm(self.mixing_weights[i, j], p='fro').item()

        return total


class UniversalGeometricNetwork(nn.Module):
    """Universal Geometric Network (UGN).

    Three-stage architecture:
    1. Encoder: Unconstrained (universal), maps X → Z with gauge choice
    2. Latent: Soft equivariant, respects bundle structure
    3. Decoder: Unconstrained (universal), interprets gauge → Y

    Achieves universal approximation + geometric consistency.

    Args:
        config: UGNConfig instance specifying architecture
    """

    def __init__(self, config: UGNConfig):
        super().__init__()
        self.config = config

        total_latent = config.n_bundles * config.bundle_dim

        # === Stage 1: Encoder ===
        self.encoder = nn.Sequential(
            SpectralLinear(config.input_dim, total_latent),
            nn.GELU(),
            SpectralLinear(total_latent, total_latent),
            nn.GELU(),
        )

        # === Stage 2: Latent Dynamics ===
        self.latent_layers = nn.ModuleList([
            SoftEquivariantLayer(
                n_bundles=config.n_bundles,
                bundle_dim=config.bundle_dim,
            )
            for _ in range(config.n_latent_layers)
        ])

        # === Stage 3: Decoder ===
        self.decoder = nn.Sequential(
            SpectralLinear(total_latent, total_latent),
            nn.GELU(),
            SpectralLinear(total_latent, config.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, input_dim]
        Returns:
            y: [B, output_dim]
        """
        # Encode
        z = self.encoder(x)  # [B, n_bundles * bundle_dim]

        # Reshape to bundle structure
        B = z.shape[0]
        z = z.view(B, self.config.n_bundles, self.config.bundle_dim)

        # Latent dynamics
        for layer in self.latent_layers:
            z = layer(z)

        # Flatten back
        z = z.view(B, -1)

        # Decode
        y = self.decoder(z)

        return y

    def regularization_loss(self) -> torch.Tensor:
        """L1 regularization on mixing pathway weights.

        Returns:
            Scalar loss [dimensionless]
        """
        loss = torch.tensor(0.0, device=next(self.parameters()).device)

        for layer in self.latent_layers:
            loss = loss + layer.l1_loss()

        return self.config.lambda_l1 * loss

    def equivariance_violation(self, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Measure equivariance violation: ||D(Rz) - RD(z)||².

        Samples random rotations R ∈ SO(d_b) and measures violation.

        Args:
            z: [B, n_bundles, bundle_dim] latent state. If None, uses random.

        Returns:
            Scalar violation [dimensionless]
        """
        if z is None:
            # Sample random latent
            B = 8
            z = torch.randn(
                B,
                self.config.n_bundles,
                self.config.bundle_dim,
                device=next(self.parameters()).device,
            )

        B, n_b, d_b = z.shape
        device = z.device

        # Sample independent rotation R_i ∈ SO(d_b) for each bundle
        # (product group equivariance: ∏_i SO(d_b)_i)
        Rs = []
        for _ in range(n_b):
            A = torch.randn(d_b, d_b, device=device)
            Q, _ = torch.linalg.qr(A)
            # Ensure det(Q) = 1 (special orthogonal)
            if torch.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            Rs.append(Q)

        # Apply per-bundle rotations
        z_rot = z.clone()
        for i in range(n_b):
            z_rot[:, i, :] = z[:, i, :] @ Rs[i].T  # [B, d_b] @ [d_b, d_b] = [B, d_b]

        # Compute D(Rz)
        z_out_1 = z_rot
        for layer in self.latent_layers:
            z_out_1 = layer(z_out_1)

        # Compute RD(z) with per-bundle rotations
        z_out_2 = z
        for layer in self.latent_layers:
            z_out_2 = layer(z_out_2)
        # Apply per-bundle rotations to output
        z_out_2_rot = z_out_2.clone()
        for i in range(n_b):
            z_out_2_rot[:, i, :] = z_out_2[:, i, :] @ Rs[i].T

        # Measure violation: ||D(Rz) - RD(z)||²
        violation = F.mse_loss(z_out_1, z_out_2_rot)

        return violation

    def total_loss(
        self,
        x: torch.Tensor,
        y_target: torch.Tensor,
        task_loss_fn: nn.Module = nn.MSELoss(),
    ) -> dict:
        """Compute total loss with regularization.

        Args:
            x: [B, input_dim] inputs
            y_target: [B, output_dim] targets
            task_loss_fn: Loss function for task

        Returns:
            Dictionary with:
                - total: Total loss
                - task: Task loss
                - l1: L1 regularization
                - equiv: Equivariance violation (if λ_equiv > 0)
        """
        # Forward pass
        y_pred = self(x)

        # Task loss
        loss_task = task_loss_fn(y_pred, y_target)

        # Regularization losses
        loss_l1 = self.regularization_loss()

        loss_dict = {
            'task': loss_task,
            'l1': loss_l1,
        }

        total = loss_task + loss_l1

        # Optional equivariance penalty
        if self.config.lambda_equiv > 0:
            loss_equiv = self.equivariance_violation()
            loss_dict['equiv'] = loss_equiv
            total = total + self.config.lambda_equiv * loss_equiv

        loss_dict['total'] = total

        return loss_dict

    def get_diagnostics(self) -> dict:
        """Diagnostic metrics for monitoring.

        Returns:
            Dictionary with:
                - mixing_strength: Total mixing pathway strength per layer
                - spectral_norms: Spectral norms of all linear layers
                - equiv_violation: Current equivariance violation
        """
        diagnostics = {}

        # Mixing strength per layer
        mixing_strengths = []
        for i, layer in enumerate(self.latent_layers):
            mixing_strengths.append(layer.mixing_strength())
        diagnostics['mixing_strength'] = mixing_strengths

        # Spectral norms (should all be ≤ 1)
        spectral_norms = []
        for name, module in self.named_modules():
            if isinstance(module, SpectralLinear):
                W = module.linear.weight
                sigma_max = torch.linalg.matrix_norm(W, ord=2).item()
                spectral_norms.append((name, sigma_max))
        diagnostics['spectral_norms'] = spectral_norms

        # Equivariance violation
        with torch.no_grad():
            diagnostics['equiv_violation'] = self.equivariance_violation().item()

        return diagnostics


# === Example usage ===
if __name__ == "__main__":
    # Configuration
    config = UGNConfig(
        input_dim=784,  # 28×28 MNIST
        output_dim=10,  # 10 classes
        n_bundles=4,
        bundle_dim=16,
        n_latent_layers=4,
        lambda_l1=0.01,
        lambda_equiv=0.0,
    )

    # Create model
    model = UniversalGeometricNetwork(config)

    # Dummy data
    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))
    y_onehot = F.one_hot(y, num_classes=10).float()

    # Forward + loss
    loss_dict = model.total_loss(x, y_onehot, task_loss_fn=nn.MSELoss())

    print("Losses:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")

    # Diagnostics
    diag = model.get_diagnostics()
    print(f"\nMixing strengths: {diag['mixing_strength']}")
    print(f"Equiv violation: {diag['equiv_violation']:.4f}")
```

### Theoretical Guarantees

We now prove the key properties of the Universal Geometric Network.

:::{prf:theorem} Universal Approximation
:label: thm-ugn-universal-approximation

Let $\mathcal{A}_{\text{UGN}}$ be a Universal Geometric Network with:
- Encoder $E: \mathcal{X} \to \mathcal{Z}$ using SpectralLinear + GELU
- Decoder $P: \mathcal{Z} \to \mathcal{Y}$ using SpectralLinear + GELU
- Latent dynamics $D = D_L \circ \cdots \circ D_1$ with soft-equivariant layers

Then for any continuous function $f: \mathcal{X} \to \mathcal{Y}$ on compact domains and any $\epsilon > 0$, there exists a choice of weights such that:

$$
\sup_{x \in \mathcal{X}} \|P(D(E(x))) - f(x)\| < \epsilon
$$

*Proof.*

**Step 1 (Encoder universality):** By the universal approximation theorem for MLPs with non-polynomial activations (Cybenko 1989, Hornik 1991), the encoder $E$ with GELU activation can approximate any continuous function $g: \mathcal{X} \to \mathcal{Z}$ to arbitrary precision on compact sets. Spectral normalization rescales but doesn't change the function class (can be compensated by adjusting subsequent layer scales).

**Step 2 (Decoder universality):** Similarly, the decoder $P$ can approximate any continuous function $h: \mathcal{Z} \to \mathcal{Y}$.

**Step 3 (Composition):** To approximate $f: \mathcal{X} \to \mathcal{Y}$, consider the decomposition:

$$
f(x) = \underbrace{P}_{\text{decoder}} \left( \underbrace{D}_{\text{latent}} \left( \underbrace{E(x)}_{\text{encoder}} \right) \right)
$$

**Strategy:** Choose:
- $E$ to approximately invert $P^{-1} \circ f$ (if $P$ were bijective)
- $D = \text{identity}$ (or close to identity)
- $P \approx f \circ E^{-1}$ (if $E$ were bijective)

More precisely:

**Sub-step 3a (Anchor points):** For a finite $\epsilon$-net $\{x_i\}_{i=1}^N$ covering $\mathcal{X}$ (exists by compactness), we need:

$$
P(D(E(x_i))) \approx f(x_i) \quad \forall i
$$

**Sub-step 3b (Encoder design):** Let the encoder map $x_i \mapsto z_i$ for arbitrary chosen $z_i \in \mathcal{Z}$ (using encoder's universality).

**Sub-step 3c (Latent passthrough):** With soft equivariance and small $\lambda_{\text{L1}}$, the mixing pathway can be trained to near-zero, making $D \approx \text{identity}$ plus small perturbations. Alternatively, with $L$ latent layers and residual connections, $D$ can implement arbitrary smooth maps via composition.

**Sub-step 3d (Decoder design):** Let the decoder satisfy $P(z_i) \approx f(x_i)$ (using decoder's universality).

**Sub-step 3e (Continuity):** By continuity of $E$, $D$, $P$ and density of $\{x_i\}$ in $\mathcal{X}$, we have:

$$
\sup_{x \in \mathcal{X}} \|P(D(E(x))) - f(x)\| \leq \sup_i \|P(D(E(x_i))) - f(x_i)\| + \underbrace{\text{continuity error}}_{\to 0 \text{ as } N \to \infty}
$$

**Step 4 (Soft equivariance doesn't restrict):** The key observation is that soft equivariance (with $\lambda_{\text{L1}}$ finite) allows the mixing pathway to activate when needed. The L1 penalty is a regularization, not a hard constraint. During training, if the task requires symmetry breaking (i.e., the target $f$ cannot be well-approximated by a strictly equivariant $D$), the optimizer will increase mixing weights, paying the L1 cost but achieving better task loss. The total loss minimization balances:

$$
\min_{\theta} \mathcal{L}_{\text{task}} + \lambda_{\text{L1}} \|W^{\text{mix}}\|_1
$$

For any fixed $\lambda_{\text{L1}} < \infty$ and $\epsilon > 0$, there exist weights achieving $\mathcal{L}_{\text{task}} < \epsilon$ (possibly with large $\|W^{\text{mix}}\|_1$).

Therefore, the UGN is a universal approximator. $\square$

**Remark (Capacity vs Expressiveness):** The theorem guarantees *existence* of approximating weights, not *learnability* via gradient descent. In practice, strong L1 regularization biases the network toward equivariant solutions, which may fail to converge to the universal approximator regime for highly non-equivariant targets. This is a feature, not a bug: the inductive bias toward geometry helps when tasks respect structure, and can be relaxed (by decreasing $\lambda_{\text{L1}}$) when necessary.
:::

:::{div} feynman-prose
Let me explain what just happened, because the proof strategy is illuminating.

The theorem says: the UGN can approximate any continuous function. Any. Not "any equivariant function"—*any* function at all. This might seem to contradict what we said earlier about strict equivariance being limiting. But it does not, and understanding why reveals the whole trick.

The key is that the encoder and decoder are *unconstrained*. They can do anything. The latent dynamics in the middle are only *softly* equivariant—the L1 penalty encourages equivariance but does not enforce it absolutely.

Here is the strategy the proof uses. Suppose you want to approximate some arbitrary function $f$. The encoder can learn to map inputs into a latent space where $f$ becomes "simple"—maybe close to the identity, maybe some easily computable transformation. The decoder then learns to extract the answer from that latent representation. The latent dynamics sit in the middle passing information through, possibly with some light processing.

The point is: the encoder and decoder do the "hard" work of representing arbitrary functions. The latent dynamics can stay close to equivariant because the hard work has been outsourced to the boundaries.

Now, you might ask: if the encoder and decoder do all the work, what is the point of the geometric structure in the middle? Great question. The answer is *inductive bias*. When the task *does* have geometric structure—when it respects rotational symmetry, when bundles should not mix unnecessarily—the L1 regularization will find that structure. The latent dynamics will stay equivariant because that is the low-cost solution. But when the task requires symmetry breaking, the mixing pathway can activate, and the network can learn whatever it needs.

The geometry is not a constraint that limits what you can learn. It is a *prior* that guides you toward structured solutions when they exist, while still allowing unstructured solutions when necessary.
:::

:::{prf:theorem} Geometric Consistency
:label: thm-ugn-geometric-consistency

Let $\mathcal{A}_{\text{UGN}}$ be a UGN with $\lambda_{\text{L1}} > 0$. Then:

1. **Approximate capacity bound:** For all layers with spectral normalization, $\|W\|_2 \leq 1$ and approximately 1-Lipschitz activations ensure:

   $$
   \|z_{\text{out}}\| \lesssim \|z_{\text{in}}\| + O(\sqrt{L})
   $$
   where $L$ is network depth (approximately non-expansive for fixed depth, consistent with $U(1)_Y$ hypercharge conservation)

2. **Bundle structure preservation:** The latent space maintains decomposition $\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i$ throughout forward pass (bundles indexed consistently)

3. **Soft equivariance:** Define the **equivariance violation** as:

   $$
   \mathcal{V}(D) = \mathbb{E}_{z \sim \mathcal{Z}, R \sim \mu_{SO(d_b)}} \|D(Rz) - RD(z)\|^2
   $$
   where $\mu_{SO(d_b)}$ is the Haar measure on $SO(d_b)$.
   Then:

   $$
   \mathcal{V}(D) \leq C \cdot \|W^{\text{mix}}\|_F^2
   $$
   for constant $C$ depending on architecture width. L1 regularization drives $\|W^{\text{mix}}\|_1 \to 0$, which implies $\|W^{\text{mix}}\|_F \to 0$, thus $\mathcal{V}(D) \to 0$.

*Proof.*

**(1) Capacity bound:** By properties of spectral norm (see Section 04, Theorem {prf:ref}`thm-spectral-preserves-hypercharge`):

$$
\|Wz\|_2 \leq \|W\|_2 \|z\|_2 \leq \|z\|_2
$$
Activations (GELU, softplus) are approximately 1-Lipschitz: for large $|x|$, both behave as $\sigma(x) \approx x$, so $\|\sigma(Wz)\|_2 \lesssim \|Wz\|_2 \leq \|z\|_2$ plus bias terms. Composing $L$ spectrally normalized layers with these activations gives:

$$
\|z_{\text{out}}\|_2 \lesssim \|z_{\text{in}}\|_2 + O(\sqrt{L})
$$
where the $O(\sqrt{L})$ term comes from accumulated biases. For fixed depth $L$, this ensures approximate capacity preservation. $\square$

**(2) Bundle structure:** The reshape operations in the forward pass (line `z.view(B, n_bundles, bundle_dim)` and back) maintain the index mapping. Each soft-equivariant layer operates on the bundle structure explicitly (indexing over $i, j \in \{1, \ldots, n_b\}$), so bundles are never "mixed" across their index—only their *contents* are transformed. $\square$

**(3) Soft equivariance bound:**

**Step 1.** Decompose the latent dynamics:

$$
D(z) = D^{\text{equiv}}(z) + D^{\text{mix}}(z)
$$

**Step 2.** The equivariant pathway satisfies $D^{\text{equiv}}(Rz) = R D^{\text{equiv}}(z)$ by construction (norm-based, Theorem {prf:ref}`thm-equivariant-function-structure`).

**Step 3.** The violation comes entirely from the mixing pathway:

$$
D(Rz) - RD(z) = \bigl[ D^{\text{equiv}}(Rz) + D^{\text{mix}}(Rz) \bigr] - R \bigl[ D^{\text{equiv}}(z) + D^{\text{mix}}(z) \bigr]
$$

$$
= \bigl[ RD^{\text{equiv}}(z) + D^{\text{mix}}(Rz) \bigr] - \bigl[ RD^{\text{equiv}}(z) + RD^{\text{mix}}(z) \bigr]
$$

$$
= D^{\text{mix}}(Rz) - RD^{\text{mix}}(z)
$$

**Step 4.** For the mixing pathway implemented as $D^{\text{mix}}(z) = \sum_{i,j} W_{ij} z_j$ (linear in $z$ for each bundle component):

$$
D^{\text{mix}}(Rz) = \sum_{ij} W_{ij} (Rz_j)
$$

$$
RD^{\text{mix}}(z) = R \sum_{ij} W_{ij} z_j
$$

The violation is bounded using operator norm arithmetic. Taking expectation over $R$ and applying Cauchy-Schwarz gives:

$$
\mathbb{E}_R \|D^{\text{mix}}(Rz) - RD^{\text{mix}}(z)\|^2 \leq C_1 \|W^{\text{mix}}\|_F^2 \|z\|^2
$$

Averaging over $z$ with $\|z\|^2$ bounded by capacity constraint:

$$
\mathcal{V}(D) \leq C \|W^{\text{mix}}\|_F^2
$$

**Step 5.** The regularization uses **group lasso** (Frobenius norm per block): $\mathcal{L}_{\text{reg}} = \lambda_{\text{L1}} \sum_{i \neq j} \|W_{ij}\|_F$ where $W_{ij}$ is the $[\text{bundle\_dim} \times \text{bundle\_dim}]$ block coupling bundles $i$ and $j$. This encourages entire blocks to be zero (sparsity at the bundle-interaction level). Since $\|W\|_F \leq \sqrt{n_{\text{blocks}}} \cdot \max_{ij} \|W_{ij}\|_F$, we have:

$$
\lambda_{\text{L1}} \sum_{ij} \|W_{ij}\|_F \to \text{large penalty} \implies \|W_{ij}\|_F \to 0 \text{ for most } (i,j) \implies \|W^{\text{mix}}\|_F \to 0 \implies \mathcal{V}(D) \to 0
$$

Therefore, group lasso regularization enforces soft equivariance with block-wise sparsity. $\square$
:::

:::{prf:proposition} Emergent Gauge Structure from Group Lasso
:label: prop-emergent-gauge-structure

Consider a UGN trained with group lasso regularization $\lambda_{\text{L1}} \sum_{i \neq j} \|W_{ij}\|_F$ (Frobenius norm per block) on a task where the true target function $f^*$ is approximately equivariant (i.e., $f^*(Rx) \approx Rf^*(x)$ for rotations $R$).

Then at convergence, the learned mixing weights $W^{\text{mix}}$ exhibit:

1. **Sparsity:** Most entries $W_{ij}^{(k\ell)}$ (mixing from bundle $j$, component $\ell$ to bundle $i$, component $k$) are driven to zero

2. **Texture zeros:** Specific cross-bundle couplings $(i, j)$ have $\|W_{ij}\|_F \approx 0$ (entire blocks zeroed out), analogous to the CKM/PMNS mixing matrices in particle physics

3. **Hierarchical structure:** Non-zero couplings organize into a hierarchy $|W_{ij}^{(1)}| \gg |W_{ij}^{(2)}| \gg \cdots$, where superscripts index components by magnitude

*Informal justification:*

**Step 1 (Group lasso induces block sparsity):** The group lasso penalty $\sum_{i \neq j} \|W_{ij}\|_F$ (sum of Frobenius norms of blocks) has a non-differentiable minimum at $W_{ij} = 0$ for each block. During gradient descent, small blocks receive gradients pushing them toward zero (block soft thresholding). If the task loss can be minimized without a particular cross-bundle coupling, group lasso drives the entire block to zero.

**Step 2 (Equivariant tasks don't need mixing):** If $f^*$ is equivariant, the optimal network architecture is strictly equivariant ($W^{\text{mix}} = 0$). The equivariant pathway $D^{\text{equiv}}$ can achieve $\mathcal{L}_{\text{task}} \approx 0$ alone. Thus:

$$
\min_{W^{\text{mix}}} \mathcal{L}_{\text{task}}(W^{\text{mix}}) + \lambda \sum_{i \neq j} \|W_{ij}\|_F
$$
has solution $W^{\text{mix}} \approx 0$.

**Step 3 (Symmetry breaking only where needed):** If $f^*$ is *mostly* equivariant but requires small symmetry breaking (e.g., $f^*(Rx) = Rf^*(x) + \epsilon(x, R)$ with $|\epsilon| \ll 1$), the optimizer activates only the minimal set of block couplings needed to capture $\epsilon$. This produces texture zeros: most bundle pairs $(i,j)$ have $\|W_{ij}\|_F \approx 0$ (entire blocks zeroed), only a few are non-zero.

**Step 4 (Hierarchy from optimization dynamics):** The group lasso penalty creates a "rich get richer" dynamic at the block level: once a coupling block $W_{ij}$ is activated (becomes non-zero), further task gradients can flow through it. Blocks that remain near zero get driven to exactly zero by group lasso. This bifurcation produces a hierarchical distribution of block coupling strengths.

**Empirical validation:** This proposition predicts measurable structure in trained models. Diagnostic node 67 (gauge invariance) from the sieve (Chapter 02) can measure $\mathcal{V}(D)$ and the sparsity pattern of $W^{\text{mix}}$. We expect to observe emergent texture zeros without hard-coding them.
:::

### Connection to BAOAB Integrator

The Universal Geometric Network integrates naturally with the Boris-BAOAB geodesic integrator from Section 05.

:::{admonition} Connection to Section 05: Latent Layers as BAOAB Steps
:class: note

Each soft-equivariant latent layer $D_\ell$ can be interpreted as implementing a discrete BAOAB step for the Lorentz-Langevin dynamics:

$$
dz^k = \underbrace{-G^{kj}\partial_j \Phi}_{\text{B: gradient}} ds + \underbrace{\beta \mathcal{F}^k{}_j \dot{z}^j}_{\text{A: Lorentz force}} ds - \underbrace{\Gamma^k_{ij}\dot{z}^i \dot{z}^j}_{\text{A: geodesic}} ds + \underbrace{\sqrt{2T} dW^k}_{\text{O: noise}}
$$

**Mapping:**

| BAOAB Component | Latent Layer Implementation |
|-----------------|---------------------------|
| **B-step** (gradient drift) | Equivariant pathway: $v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|)$ computes energy-based scales from norm MLP taking all bundle norms as input |
| **A-step** (Lorentz force) | Mixing pathway: $W_{ij} v_j$ implements cross-bundle coupling (value curl analog) |
| **A-step** (geodesic correction) | Norm-gating: $v \to v / \|v\| \cdot g(\|v\|)$ projects to manifold |
| **B-step** (second gradient) | Symmetric to first B-step (residual connection in layer) |
| **O-step** (Ornstein-Uhlenbeck thermostat) | Implicit in stochastic training (dropout, batch noise) or explicit noise injection |

**Multi-layer composition:** With $L = 4$ latent layers, the forward pass implements $L$ BAOAB steps, equivalent to integrating the Lorentz-Langevin SDE for $L \cdot \Delta s$ in computational time.

**Covariant cross-attention alternative:** For explicit world models, replace soft-equivariant layers with the covariant cross-attention module from Section {ref}`sec-covariant-cross-attention-definition`, which uses:
- Wilson lines for gauge-covariant Q/K/V projections
- Position-dependent temperature $\tau(z) = \sqrt{d_k} / \lambda(z)$ (inverse conformal factor)
- Geometric Query terms $Q = W_Q x + W_{Qz} z + W_{Q\Gamma}(z \otimes z)$ for Christoffel symbols

The UGN provides the *default* latent dynamics for bounded agents. Covariant cross-attention extends this to explicit trajectory prediction and planning.
:::



(sec-complete-implementation)=
## Complete Implementation

:::{div} feynman-prose
Now let me show you how to actually build this thing.

We have designed the Universal Geometric Network. We have proved it achieves universal approximation. We have proved it respects geometric consistency. But theory is cheap—paper accepts anything you write on it. The real question is: can you sit down at a keyboard, type in some code, train the network, and have it actually do what we claimed?

The answer is yes. And I want to show you exactly how.

This section gives you production-ready code. Not pseudocode, not hand-waving—actual PyTorch that you can copy, run, and modify. It integrates everything: the spectral linear layers from Section 04 (for capacity bounds), the soft equivariance machinery we just designed (for geometric structure), and the BAOAB integrator from Section 05 (for geodesic dynamics). You will see exactly how the pieces fit together.

I want to emphasize: this is the *engineer's* section. No more theorems. No more proofs. Just working code and the protocol to train it. You will get:

- Complete PyTorch implementations with shape annotations so you know exactly what goes where
- A training loop that balances task performance with geometric consistency
- Diagnostic signals to monitor so you know when things are working and when they are not
- The adaptive L1 scheduler that finds the right symmetry-breaking level automatically

If you have been following the mathematical development, this is where abstraction becomes concrete. If you skipped ahead looking for something you can actually run, this is your entry point. Either way, pay attention to the diagnostic code—it will tell you what the network is learning about structure, which is the whole point of the geometric setup.
:::

### Full Architecture Code

The complete UGN architecture consists of four components:

1. **Configuration dataclasses** - Bundle specifications and hyperparameters
2. **SpectralLinear** - Capacity-bounded linear layers (from Section 04 {prf:ref}`def-spectral-linear`)
3. **SoftEquivariantLayer** - Latent dynamics with soft equivariance
4. **UniversalGeometricNetwork** - Complete encoder-dynamics-decoder pipeline

We reference Section 04 primitives rather than reimplementing them.

:::{note}
:class: feynman-added
**Production Implementation:** This section provides a **complete production implementation** supporting **heterogeneous bundle dimensions** via `List[BundleConfig]`. This extends the simplified implementation in Section 5 (which assumes uniform `bundle_dim` for all bundles). Use this implementation when bundles need different dimensional capacities.
:::

Here's the minimal necessary code:

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BundleConfig:
    """Configuration for a single gauge bundle.

    Each bundle represents a fiber $V_b$ with internal SO(d_b) symmetry.

    Args:
        name: Semantic label (e.g., "charge", "color", "lepton", "yukawa")
        dim: Bundle dimension $d_b$ [dimensionless]
        semantic_role: Physical interpretation
    """
    name: str
    dim: int
    semantic_role: str = ""

@dataclass
class UGNConfig:
    """Configuration for Universal Geometric Network.

    Specifies the complete three-stage architecture:
    - Encoder: $\mathbb{R}^{d_{\text{in}}} \to \bigoplus_{i=1}^{n_b} V_i$
    - Latent: Soft equivariant dynamics on bundles
    - Decoder: $\bigoplus_{i=1}^{n_b} V_i \to \mathbb{R}^{d_{\text{out}}}$

    Args:
        input_dim: Input dimension $d_{\text{in}}$ [dimensionless]
        output_dim: Output dimension $d_{\text{out}}$ [dimensionless]
        bundles: List of bundle specifications
        n_latent_layers: Number of soft equivariant layers $L$ [dimensionless]
        encoder_hidden_dim: Hidden dimension in encoder [dimensionless]
        decoder_hidden_dim: Hidden dimension in decoder [dimensionless]
        lambda_l1: L1 regularization strength $\lambda_{\text{L1}}$ [dimensionless]
        lambda_equiv: Equivariance violation penalty $\lambda_{\text{equiv}}$ [dimensionless]
        use_spectral_norm: Apply spectral normalization (capacity bound) [bool]
    """
    input_dim: int
    output_dim: int
    bundles: List[BundleConfig]
    n_latent_layers: int = 4
    encoder_hidden_dim: int = 256
    decoder_hidden_dim: int = 256
    lambda_l1: float = 0.01
    lambda_equiv: float = 0.0  # Start at 0, increase if needed
    use_spectral_norm: bool = True

    @property
    def n_bundles(self) -> int:
        """Number of gauge bundles $n_b$."""
        return len(self.bundles)

    @property
    def total_latent_dim(self) -> int:
        """Total latent dimension $\sum_{i=1}^{n_b} d_i$."""
        return sum(b.dim for b in self.bundles)

    @property
    def bundle_dims(self) -> List[int]:
        """List of bundle dimensions $[d_1, d_2, \ldots, d_{n_b}]$."""
        return [b.dim for b in self.bundles]


# ============================================================================
# Spectral Linear Layer (Section 04 Primitive)
# ============================================================================

class SpectralLinear(nn.Module):
    """Linear layer with spectral normalization: $\sigma_{\max}(W) \leq 1$.

    Implements {prf:ref}`def-spectral-linear` from Section 04.
    Ensures capacity bound: no unbounded amplification of input signals.

    Args:
        in_features: Input dimension $d_{\text{in}}$ [dimensionless]
        out_features: Output dimension $d_{\text{out}}$ [dimensionless]
        bias: Include bias term [bool]

    Forward:
        x: Input tensor [B, d_in]

    Returns:
        y: Output tensor [B, d_out] with $\|y\| \leq \|x\|$ (capacity bound)
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Apply spectral normalization (PyTorch native implementation)
        self.linear = nn.utils.spectral_norm(self.linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral-normalized linear transformation."""
        return self.linear(x)


# ============================================================================
# Soft Equivariant Layer (Core Latent Dynamics)
# ============================================================================

class SoftEquivariantLayer(nn.Module):
    """Soft equivariant latent dynamics layer.

    Implements the key idea: decompose dynamics into equivariant and mixing pathways,
    with L1 regularization discovering emergent structure in the mixing weights.

    Forward:
        z = equivariant_path(z) + mixing_path(z)

    - **Equivariant path**: $f_i(z) = v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|)$
      Uses only norms → strictly equivariant under $\prod_i SO(d_i)$

    - **Mixing path**: $M(z) = \sum_{i,j} W_{ij} v_j$
      Cross-bundle mixing with L1 penalty → emergent texture zeros

    Args:
        bundle_dims: List of bundle dimensions $[d_1, \ldots, d_{n_b}]$ [dimensionless]
        hidden_dim: Hidden dimension for norm MLP $\phi$ [dimensionless]
        use_spectral_norm: Apply spectral normalization [bool]

    Forward:
        z: Latent tensor [B, sum(bundle_dims)]

    Returns:
        z_out: Updated latent [B, sum(bundle_dims)]
    """
    def __init__(
        self,
        bundle_dims: List[int],
        hidden_dim: int = 64,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        self.bundle_dims = bundle_dims
        self.n_bundles = len(bundle_dims)
        self.total_dim = sum(bundle_dims)
        self.hidden_dim = hidden_dim

        # Equivariant pathway: norms → scales
        # MLP: R^{n_b} → R^{n_b}, input = [||v_1||, ..., ||v_{n_b}||]
        LinearLayer = SpectralLinear if use_spectral_norm else nn.Linear

        self.norm_mlp = nn.Sequential(
            LinearLayer(self.n_bundles, hidden_dim, bias=True),
            nn.GELU(),
            LinearLayer(hidden_dim, hidden_dim, bias=True),
            nn.GELU(),
            LinearLayer(hidden_dim, self.n_bundles, bias=False)
        )

        # Mixing pathway: cross-bundle interactions (L1 regularized)
        # W_{ij}: V_j → V_i, shape [n_bundles, n_bundles, d_i, d_j]
        # Initialize small to bias toward equivariant pathway
        self.mixing_weights = nn.ParameterList([
            nn.ParameterList([
                nn.Parameter(torch.randn(self.bundle_dims[i], self.bundle_dims[j]) * 0.01)
                for j in range(self.n_bundles)
            ])
            for i in range(self.n_bundles)
        ])

        # Gate bias for mixing pathway (learnable interpolation)
        self.gate_bias = nn.Parameter(torch.zeros(self.n_bundles))

    def split_bundles(self, z: torch.Tensor) -> List[torch.Tensor]:
        """Split latent z into bundle components [v_1, ..., v_{n_b}].

        Args:
            z: Latent tensor [B, sum(d_i)]

        Returns:
            List of [v_1, ..., v_{n_b}] where v_i has shape [B, d_i]
        """
        bundles = []
        offset = 0
        for dim in self.bundle_dims:
            bundles.append(z[:, offset:offset+dim])
            offset += dim
        return bundles

    def cat_bundles(self, bundles: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate bundle components back to latent z.

        Args:
            bundles: List of [v_1, ..., v_{n_b}] where v_i has shape [B, d_i]

        Returns:
            z: Latent tensor [B, sum(d_i)]
        """
        return torch.cat(bundles, dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply soft equivariant dynamics: z → z'.

        Args:
            z: Latent tensor [B, sum(d_i)]

        Returns:
            z_out: Updated latent [B, sum(d_i)]
        """
        bundles = self.split_bundles(z)  # [v_1, ..., v_{n_b}]

        # ====================================================================
        # Equivariant Pathway: v_i → v_i · φ_i(||v_1||, ..., ||v_{n_b}||)
        # ====================================================================

        # Compute norms: ||v_i|| for each bundle
        norms = torch.stack([
            torch.norm(v, dim=-1) + 1e-8  # [B]
            for v in bundles
        ], dim=-1)  # [B, n_bundles]

        # Norm MLP: [||v_1||, ..., ||v_{n_b}||] → [s_1, ..., s_{n_b}]
        scales = self.norm_mlp(norms)  # [B, n_bundles]

        # Apply isotropic scaling: v_i → v_i · s_i
        equivariant_outputs = [
            bundles[i] * scales[:, i:i+1]  # [B, d_i] * [B, 1] = [B, d_i]
            for i in range(self.n_bundles)
        ]

        # ====================================================================
        # Mixing Pathway: M(z) = Σ_{i,j} W_{ij} v_j (L1 regularized)
        # ====================================================================

        mixing_outputs = []
        for i in range(self.n_bundles):
            # For bundle i, compute: Σ_j W_{ij} v_j
            mixed = sum(
                F.linear(bundles[j], self.mixing_weights[i][j])  # [B, d_i]
                for j in range(self.n_bundles)
            )
            mixing_outputs.append(mixed)

        # ====================================================================
        # Combine Pathways: z_out = equivariant + gate * mixing
        # ====================================================================

        # Learnable gate per bundle (sigmoid for [0, 1] range)
        gates = torch.sigmoid(self.gate_bias)  # [n_bundles]

        combined_bundles = [
            equivariant_outputs[i] + gates[i] * mixing_outputs[i]
            for i in range(self.n_bundles)
        ]

        z_out = self.cat_bundles(combined_bundles)

        # Residual connection for stable training
        return z + z_out

    def l1_loss(self) -> torch.Tensor:
        """L1 penalty on mixing pathway weights.

        Encourages sparsity → emergent texture zeros and hierarchical structure.

        Returns:
            L1 norm of all mixing weights [scalar]
        """
        total_l1 = sum(
            torch.sum(torch.abs(self.mixing_weights[i][j]))
            for i in range(self.n_bundles)
            for j in range(self.n_bundles)
        )
        return total_l1

    def mixing_strength(self) -> float:
        """Total strength of mixing pathway (diagnostic).

        Returns:
            ||W||_F: Frobenius norm of all mixing weights [dimensionless]
        """
        total_norm_sq = sum(
            torch.sum(self.mixing_weights[i][j] ** 2)
            for i in range(self.n_bundles)
            for j in range(self.n_bundles)
        )
        return torch.sqrt(total_norm_sq).item()


# ============================================================================
# Universal Geometric Network (Complete Architecture)
# ============================================================================

class UniversalGeometricNetwork(nn.Module):
    """Universal Geometric Network (UGN).

    Three-stage architecture achieving both universal approximation and geometric consistency:

    1. **Encoder** (unconstrained, universal):
       $\mathbb{R}^{d_{\text{in}}} \to \bigoplus_i V_i$
       - Spectral normalization for capacity bound
       - Chooses gauge for latent representation

    2. **Latent Dynamics** (soft equivariant):
       $\bigoplus_i V_i \to \bigoplus_i V_i$
       - Gauge-covariant evolution respecting bundle structure
       - L1 regularization discovers emergent symmetry breaking

    3. **Decoder** (unconstrained, universal):
       $\bigoplus_i V_i \to \mathbb{R}^{d_{\text{out}}}$
       - Spectral normalization for capacity bound
       - Interprets gauge-dependent latent → gauge-invariant output

    **Key theorems:**
    - {prf:ref}`thm-ugn-universal-approximation`: UGN is universal
    - {prf:ref}`thm-ugn-geometric-consistency`: Respects bundle structure
    - {prf:ref}`prop-emergent-gauge-structure`: L1 discovers texture zeros

    Args:
        config: UGNConfig specifying architecture

    Forward:
        x: Input tensor [B, d_in]

    Returns:
        y: Output tensor [B, d_out]
    """
    def __init__(self, config: UGNConfig):
        super().__init__()
        self.config = config

        LinearLayer = SpectralLinear if config.use_spectral_norm else nn.Linear

        # ====================================================================
        # Stage 1: Encoder (unconstrained → universal approximation)
        # ====================================================================

        self.encoder = nn.Sequential(
            LinearLayer(config.input_dim, config.encoder_hidden_dim),
            nn.GELU(),
            LinearLayer(config.encoder_hidden_dim, config.encoder_hidden_dim),
            nn.GELU(),
            LinearLayer(config.encoder_hidden_dim, config.total_latent_dim),
        )

        # ====================================================================
        # Stage 2: Latent Dynamics (soft equivariant)
        # ====================================================================

        self.latent_layers = nn.ModuleList([
            SoftEquivariantLayer(
                bundle_dims=config.bundle_dims,
                hidden_dim=64,
                use_spectral_norm=config.use_spectral_norm
            )
            for _ in range(config.n_latent_layers)
        ])

        # ====================================================================
        # Stage 3: Decoder (unconstrained → universal approximation)
        # ====================================================================

        self.decoder = nn.Sequential(
            LinearLayer(config.total_latent_dim, config.decoder_hidden_dim),
            nn.GELU(),
            LinearLayer(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.GELU(),
            LinearLayer(config.decoder_hidden_dim, config.output_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder: X → Z (chooses gauge)."""
        return self.encoder(x)

    def dynamics(self, z: torch.Tensor) -> torch.Tensor:
        """Latent dynamics: Z → Z' (soft equivariant evolution)."""
        for layer in self.latent_layers:
            z = layer(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decoder: Z → Y (interprets gauge)."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: X → Z → Z' → Y."""
        z = self.encode(x)
        z = self.dynamics(z)
        y = self.decode(z)
        return y

    def regularization_loss(self) -> torch.Tensor:
        """L1 regularization on mixing pathway weights.

        Returns:
            Total L1 penalty summed over all latent layers [scalar]
        """
        total_l1 = sum(layer.l1_loss() for layer in self.latent_layers)
        return total_l1

    def equivariance_violation(
        self,
        z: Optional[torch.Tensor] = None,
        n_samples: int = 16
    ) -> torch.Tensor:
        """Measure product group equivariance violation in latent dynamics.

        Samples independent rotations R_i ∈ SO(d_i) for each bundle and computes:
        $$\epsilon(z, \{R_i\}) = \|D(Rz) - RD(z)\|^2 \text{ where } R = \bigoplus_i R_i$$

        Tests equivariance under ∏_i SO(d_i) (product of per-bundle rotations).

        Args:
            z: Latent tensor [B, sum(d_i)]. If None, samples random z.
            n_samples: Number of random rotation samples

        Returns:
            Mean equivariance violation [scalar]
        """
        if z is None:
            # Sample random latent
            z = torch.randn(1, self.config.total_latent_dim, device=next(self.parameters()).device)

        violations = []

        for _ in range(n_samples):
            # Sample independent rotation R_i for each bundle
            Rs = []
            for d_b in self.config.bundle_dims:
                # Sample random SO(d_b) rotation (via QR decomposition)
                A = torch.randn(d_b, d_b, device=z.device)
                Q, _ = torch.linalg.qr(A)
                # Ensure det(Q) = 1 (special orthogonal)
                if torch.det(Q) < 0:
                    Q[:, 0] = -Q[:, 0]
                Rs.append(Q)

            # Apply per-bundle rotations to input
            z_rotated = z.clone()
            for i, (d_b, R) in enumerate(zip(self.config.bundle_dims, Rs)):
                offset = sum(self.config.bundle_dims[:i])
                z_rotated[:, offset:offset+d_b] = z[:, offset:offset+d_b] @ R.T

            # Forward through dynamics
            z_out = self.dynamics(z)
            z_rotated_out = self.dynamics(z_rotated)

            # Apply per-bundle rotations to output
            z_out_rotated = z_out.clone()
            for i, (d_b, R) in enumerate(zip(self.config.bundle_dims, Rs)):
                offset = sum(self.config.bundle_dims[:i])
                z_out_rotated[:, offset:offset+d_b] = z_out[:, offset:offset+d_b] @ R.T

            # Compute violation: ||D(Rz) - RD(z)||²
            violation = torch.norm(z_rotated_out - z_out_rotated) ** 2
            violations.append(violation)

        return torch.mean(torch.stack(violations))

    def total_loss(
        self,
        x: torch.Tensor,
        y_target: torch.Tensor,
        task_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> dict:
        """Compute total loss with regularization.

        $$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_{\text{L1}} \mathcal{L}_{\text{L1}} + \lambda_{\text{equiv}} \mathcal{L}_{\text{equiv}}$$

        Args:
            x: Input [B, d_in]
            y_target: Target output [B, d_out]
            task_loss_fn: Task-specific loss (e.g., MSE, cross-entropy)

        Returns:
            Dictionary with keys:
                - 'total': Total loss (for backprop)
                - 'task': Task loss
                - 'l1': L1 regularization
                - 'equiv': Equivariance violation (if λ_equiv > 0)
        """
        # Forward pass
        y_pred = self.forward(x)

        # Task loss
        loss_task = task_loss_fn(y_pred, y_target)

        # L1 regularization
        loss_l1 = self.regularization_loss()

        # Total loss
        loss_total = loss_task + self.config.lambda_l1 * loss_l1

        result = {
            'total': loss_total,
            'task': loss_task.item(),
            'l1': loss_l1.item(),
        }

        # Optional: equivariance violation penalty
        if self.config.lambda_equiv > 0:
            z = self.encode(x)
            loss_equiv = self.equivariance_violation(z)
            loss_total = loss_total + self.config.lambda_equiv * loss_equiv
            result['equiv'] = loss_equiv.item()
            result['total'] = loss_total

        return result

    def get_diagnostics(self) -> dict:
        """Get diagnostic metrics for monitoring training.

        Returns:
            Dictionary with diagnostic values:
                - 'mixing_strength': Total mixing pathway strength ||W||_F
                - 'gate_values': Interpolation gates per bundle
                - 'bundle_norms': Typical norm per bundle (run on sample input)
        """
        diagnostics = {
            'mixing_strength': sum(layer.mixing_strength() for layer in self.latent_layers),
            'gate_values': [
                torch.sigmoid(layer.gate_bias).detach().cpu().numpy()
                for layer in self.latent_layers
            ],
        }
        return diagnostics
```

**Key implementation details:**

1. **Spectral normalization** (optional but recommended): Ensures $\sigma_{\max}(W) \leq 1$, providing capacity bound as in {prf:ref}`def-spectral-linear`.

2. **Residual connections** in `SoftEquivariantLayer`: Stabilizes training and enables deeper latent dynamics.

3. **Learnable gates**: `gate_bias` parameters allow network to learn optimal interpolation between equivariant and mixing pathways.

4. **L1 penalty**: Applied to mixing weights (not equivariant pathway), encouraging emergent sparsity.

### Integration with BAOAB Dynamics

The soft equivariant layer is **directly compatible** with the BAOAB integrator from Section 05. Here's how the connection works:

:::{prf:observation} BAOAB Structure in Soft Equivariant Layers
:label: obs-baoab-soft-equiv

Each `SoftEquivariantLayer` forward pass can be interpreted as a single BAOAB integration step:

**B (Momentum update):** Equivariant pathway computes geodesic velocity
$$v_i \to v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|)$$

**A (Position update, first half):** Mixing pathway introduces cross-bundle coupling
$$v_i \to v_i + \sum_j W_{ij} v_j$$

**O (Ornstein-Uhlenbeck thermostat):** Implicit in activation nonlinearity (GELU in norm MLP)

**A (Position update, second half):** Residual connection $z_{\text{out}} = z + \Delta z$

**B (Momentum update):** Next layer repeats

**Correspondence to Section 05:**
- {prf:ref}`def-baoab-attention-heads`: Splitting scheme for Hamiltonian dynamics
- {prf:ref}`def-covariant-qkv-projections`: Multi-head attention = parallel BAOAB chains
:::

To extend UGN to **explicit world modeling** with covariant cross-attention (Section {ref}`sec-covariant-cross-attention-architecture`):

:::{warning}
**Pseudocode - Not Runnable:** The following code references primitives from Section 04 (`WilsonLine`, `CovariantAttention`, `GeometricQuery`) that are **not implemented here**. This serves as a **design template** for integrating covariant attention into UGN. For production use, you must either:
1. Implement these primitives following Section 04 definitions, or
2. Use this as pseudocode for integration with existing gauge-covariant libraries.
:::

```python
# NOTE: This code requires Section 04 primitives (WilsonLine, CovariantAttention, GeometricQuery)
# These are referenced but not implemented here. For production use, either:
# 1. Implement these primitives following Section 04 definitions, or
# 2. Use this as pseudocode for integration with existing gauge-covariant libraries

from fragile.primitives import (  # Assumes primitives are available
    WilsonLine,
    CovariantAttention,
    GeometricQuery
)

class CovariantAttentionLayer(nn.Module):
    """Covariant cross-attention for world modeling.

    Replaces SoftEquivariantLayer for tasks requiring explicit trajectory prediction.
    Implements multi-head attention with:
    - Wilson lines for gauge-covariant Q/K/V projections
    - Position-dependent temperature τ(z) = √d_k / λ(z)
    - Geometric Query terms with Christoffel symbols

    See Ch. 05 {ref}`sec-covariant-cross-attention-definition` for details.
    """
    def __init__(
        self,
        bundle_dims: List[int],
        n_heads: int = 4,
        use_wilson_lines: bool = True
    ):
        super().__init__()
        self.bundle_dims = bundle_dims
        self.n_heads = n_heads

        # Multi-head covariant attention (one per bundle)
        self.attention_heads = nn.ModuleList([
            CovariantAttention(
                d_bundle=d_b,
                n_heads=n_heads,
                use_wilson_lines=use_wilson_lines
            )
            for d_b in bundle_dims
        ])

    def split_bundles(self, z: torch.Tensor) -> List[torch.Tensor]:
        """Split latent z into bundle components [v_1, ..., v_{n_b}].

        Args:
            z: Latent tensor [B, sum(d_i)]

        Returns:
            List of [v_1, ..., v_{n_b}] where v_i has shape [B, d_i]
        """
        bundles = []
        offset = 0
        for dim in self.bundle_dims:
            bundles.append(z[:, offset:offset+dim])
            offset += dim
        return bundles

    def cat_bundles(self, bundles: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate bundle components back to latent z.

        Args:
            bundles: List of [v_1, ..., v_{n_b}] where v_i has shape [B, d_i]

        Returns:
            z: Latent tensor [B, sum(d_i)]
        """
        return torch.cat(bundles, dim=-1)

    def forward(self, z: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply covariant cross-attention.

        Args:
            z: Latent [B, sum(d_i)]
            context: Optional context for cross-attention [B, T, sum(d_i)]

        Returns:
            z_out: Attended latent [B, sum(d_i)]
        """
        bundles = self.split_bundles(z)

        attended_bundles = []
        for i, (v, attn) in enumerate(zip(bundles, self.attention_heads)):
            # For self-attention, context = z
            # For cross-attention, extract bundle from context
            ctx = context if context is not None else v.unsqueeze(1)
            v_out = attn(v, ctx)  # [B, d_i]
            attended_bundles.append(v_out)

        return self.cat_bundles(attended_bundles)
```

**When to use each:**

| Architecture | Use Case | Reference |
|-------------|----------|-----------|
| `SoftEquivariantLayer` | Default latent dynamics, implicit world model | This chapter |
| `CovariantAttentionLayer` | Explicit trajectory prediction, planning | Section 05 {ref}`sec-covariant-cross-attention-architecture` |

**Multi-stage pipelines:** For complex tasks, combine both:
1. Encoder → latent Z
2. SoftEquivariantLayer (×2) for geometric regularization
3. CovariantAttentionLayer for trajectory rollout
4. SoftEquivariantLayer (×2) for policy extraction
5. Decoder → action Y

### Training Protocol

The complete training loop balances three objectives:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{L1}}(t) \cdot \mathcal{L}_{\text{L1}} + \lambda_{\text{equiv}}(t) \cdot \mathcal{L}_{\text{equiv}}$$

where:
- $\mathcal{L}_{\text{task}}$: Task-specific loss (MSE, cross-entropy, etc.)
- $\mathcal{L}_{\text{L1}} = \sum_{i,j} \|W_{ij}\|_1$: L1 penalty on mixing weights
- $\mathcal{L}_{\text{equiv}} = \mathbb{E}_{R \sim SO(d_b)} \|D(Rz) - RD(z)\|^2$: Equivariance violation

The regularization coefficients $\lambda_{\text{L1}}(t)$ and $\lambda_{\text{equiv}}(t)$ are **scheduled** during training.

#### L1 Regularization Schedule

:::{prf:definition} Adaptive L1 Schedule
:label: def-adaptive-l1-schedule

The L1 regularization strength adapts based on current equivariance violation:

$$\lambda_{\text{L1}}(t+1) = \lambda_{\text{L1}}(t) \cdot \left(1 + \alpha \cdot (\epsilon(t) - \epsilon_{\text{target}})\right)$$

where:
- $\epsilon(t) = \mathcal{L}_{\text{equiv}}(t)$: Current equivariance violation
- $\epsilon_{\text{target}} \approx 0.22$ nat/step: Proposed target violation (empirical; to be validated)
- $\alpha \in [0.01, 0.1]$: Learning rate for schedule

**Strategy:**
- If $\epsilon(t) > \epsilon_{\text{target}}$: Increase $\lambda_{\text{L1}}$ (more sparsity)
- If $\epsilon(t) < \epsilon_{\text{target}}$: Decrease $\lambda_{\text{L1}}$ (more expressiveness)

This implements **self-tuning** toward the natural symmetry-breaking scale.
:::

Implementation:

```python
class AdaptiveL1Scheduler:
    """Adaptive L1 regularization schedule.

    Adjusts λ_L1 to target a specific equivariance violation level.

    Args:
        initial_lambda: Initial λ_L1 value [dimensionless]
        target_violation: Target ε (typically 0.22 nat/step) [nat/step]
        learning_rate: Schedule update rate α [dimensionless]
        min_lambda: Minimum λ_L1 to prevent collapse [dimensionless]
        max_lambda: Maximum λ_L1 to prevent over-sparsity [dimensionless]
    """
    def __init__(
        self,
        initial_lambda: float = 0.01,
        target_violation: float = 0.22,
        learning_rate: float = 0.05,
        min_lambda: float = 1e-4,
        max_lambda: float = 1.0
    ):
        self.lambda_l1 = initial_lambda
        self.target_violation = target_violation
        self.alpha = learning_rate
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

        # History for diagnostics
        self.history = {
            'lambda_l1': [initial_lambda],
            'violation': [],
        }

    def step(self, current_violation: float) -> float:
        """Update λ_L1 based on current equivariance violation.

        Args:
            current_violation: Current ε value [nat/step]

        Returns:
            Updated λ_L1 [dimensionless]
        """
        # Error signal: positive if violation too high
        error = current_violation - self.target_violation

        # Adaptive update: increase λ when violation > target
        self.lambda_l1 *= (1 + self.alpha * error)

        # Clamp to valid range
        self.lambda_l1 = max(self.min_lambda, min(self.max_lambda, self.lambda_l1))

        # Record history
        self.history['lambda_l1'].append(self.lambda_l1)
        self.history['violation'].append(current_violation)

        return self.lambda_l1


def train_ugn(
    model: UniversalGeometricNetwork,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-3,
    task_loss_fn: Callable = nn.MSELoss(),
    use_adaptive_l1: bool = True,
    device: str = 'cuda'
) -> dict:
    """Complete training protocol for Universal Geometric Network.

    Args:
        model: UGN instance
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of training epochs
        lr: Learning rate
        task_loss_fn: Task-specific loss function
        use_adaptive_l1: Use adaptive L1 schedule (recommended)
        device: 'cuda' or 'cpu'

    Returns:
        Dictionary with training history and final diagnostics:
            - 'train_loss': List[float] - Training task loss per epoch
            - 'val_loss': List[float] - Validation task loss per epoch
            - 'l1_loss': List[float] - L1 regularization per epoch
            - 'equiv_violation': List[float] - Equivariance violation per epoch
            - 'lambda_l1': List[float] - Adaptive λ_L1 values per epoch
            - 'mixing_strength': List[float] - Total mixing pathway strength per epoch
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Learning rate schedule: cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Adaptive L1 schedule (optional but recommended)
    l1_scheduler = AdaptiveL1Scheduler() if use_adaptive_l1 else None

    history = {
        'train_loss': [],
        'val_loss': [],
        'l1_loss': [],
        'equiv_violation': [],
        'lambda_l1': [],
        'mixing_strength': [],
    }

    for epoch in range(n_epochs):
        # ====================================================================
        # Training Phase
        # ====================================================================
        model.train()
        epoch_losses = {'total': 0.0, 'task': 0.0, 'l1': 0.0, 'equiv': 0.0}

        for batch_idx, (x, y_target) in enumerate(train_loader):
            x = x.to(device)
            y_target = y_target.to(device)

            optimizer.zero_grad()

            # Forward pass with all loss components
            losses = model.total_loss(x, y_target, task_loss_fn)

            # Backward pass
            losses['total'].backward()

            # Gradient clipping (optional, helps stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Accumulate losses
            # Note: 'total' is Tensor (for backward), others are float (from .item())
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key] if isinstance(losses[key], float) else losses[key].item()

        # Average over batches
        n_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches

        # ====================================================================
        # Validation Phase
        # ====================================================================
        model.eval()
        val_loss = 0.0
        val_equiv_violation = 0.0

        with torch.no_grad():
            for x, y_target in val_loader:
                x = x.to(device)
                y_target = y_target.to(device)

                y_pred = model(x)
                val_loss += task_loss_fn(y_pred, y_target).item()

                # Measure equivariance violation on validation set
                # Note: This runs dynamics forward pass twice (expensive diagnostic)
                z = model.encode(x)
                val_equiv_violation += model.equivariance_violation(z).item()

        val_loss /= len(val_loader)
        val_equiv_violation /= len(val_loader)

        # ====================================================================
        # Schedule Updates
        # ====================================================================

        # Update learning rate
        scheduler.step()

        # Update L1 regularization (if adaptive)
        if l1_scheduler is not None:
            new_lambda_l1 = l1_scheduler.step(val_equiv_violation)
            model.config.lambda_l1 = new_lambda_l1

        # ====================================================================
        # Diagnostics and Logging
        # ====================================================================

        diagnostics = model.get_diagnostics()

        history['train_loss'].append(epoch_losses['task'])
        history['val_loss'].append(val_loss)
        history['l1_loss'].append(epoch_losses['l1'])
        history['equiv_violation'].append(val_equiv_violation)
        history['lambda_l1'].append(model.config.lambda_l1)
        history['mixing_strength'].append(diagnostics['mixing_strength'])

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"  Train Loss: {epoch_losses['task']:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  L1 Loss: {epoch_losses['l1']:.4f}")
            print(f"  Equiv Violation: {val_equiv_violation:.4f}")
            print(f"  λ_L1: {model.config.lambda_l1:.4f}")
            print(f"  Mixing Strength: {diagnostics['mixing_strength']:.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
            print()

    return history


# Example usage:
if __name__ == "__main__":
    # Define bundle structure (e.g., Standard Model inspired)
    bundles = [
        BundleConfig("charge", dim=16, semantic_role="U(1) charge"),
        BundleConfig("color", dim=16, semantic_role="SU(3) color"),
        BundleConfig("lepton", dim=16, semantic_role="Lepton family"),
        BundleConfig("yukawa", dim=16, semantic_role="Yukawa coupling"),
    ]

    config = UGNConfig(
        input_dim=128,
        output_dim=64,
        bundles=bundles,
        n_latent_layers=4,
        lambda_l1=0.01,
        lambda_equiv=0.0,  # Start without explicit equivariance penalty
        use_spectral_norm=True
    )

    model = UniversalGeometricNetwork(config)

    # Create dummy data loaders (replace with real data)
    train_loader = ...  # Your training data
    val_loader = ...    # Your validation data

    # Train with adaptive L1 schedule
    history = train_ugn(
        model,
        train_loader,
        val_loader,
        n_epochs=100,
        lr=1e-3,
        use_adaptive_l1=True
    )
```

#### Diagnostic Nodes for Monitoring

Following the **Sieve** diagnostic framework (Part II), monitor these key signals during training:

| Node ID | Diagnostic | Interpretation | Target Range |
|---------|-----------|----------------|--------------|
| **62** | Spectral norm $\sigma_{\max}(W)$ | Capacity bound | $\leq 1.0$ (enforced by spectral normalization) |
| **67** | Gauge invariance | Output unchanged under latent gauge transform | $< 0.01$ nat |
| **NEW** | L1 mixing strength $\|W\|_1$ | Total cross-bundle coupling | Decreases during training |
| **NEW** | Equivariance violation $\epsilon$ | Soft equivariance quality | $\approx 0.22$ nat/step (proposed empirical target) |
| **NEW** | Mixing gate values | Learned interpolation per bundle | $[0, 1]$ (monitor which bundles use mixing) |
| **NEW** | Texture zeros | Number of $\|W_{ij}\| < 10^{-3}$ entries | Increases with L1 (emergent structure) |

**Diagnostic code:**

```python
def compute_diagnostics(model: UniversalGeometricNetwork, x_sample: torch.Tensor) -> dict:
    """Compute comprehensive diagnostics for monitoring.

    Args:
        model: UGN instance
        x_sample: Sample input for diagnostics [B, d_in]

    Returns:
        Dictionary with all diagnostic values
    """
    model.eval()
    diagnostics = {}

    with torch.no_grad():
        # Node 62: Spectral norm (should be ≤ 1.0 if using spectral normalization)
        # Note: For SpectralLinear, module.weight returns the normalized weight
        spectral_norms = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight  # Normalized weight if spectral_norm applied
                sigma_max = torch.linalg.matrix_norm(W, ord=2).item()
                spectral_norms.append(sigma_max)
        diagnostics['node_62_spectral_norm_max'] = max(spectral_norms)
        diagnostics['node_62_spectral_norm_mean'] = sum(spectral_norms) / len(spectral_norms)

        # Node 67: Gauge invariance
        # Test: random gauge transform in latent → output should be unchanged
        z = model.encode(x_sample)
        y_original = model.decode(z)

        # Apply random bundle-wise rotation
        bundles = model.latent_layers[0].split_bundles(z)
        rotated_bundles = []
        for i, v in enumerate(bundles):
            d_b = model.config.bundle_dims[i]
            R = torch.linalg.qr(torch.randn(d_b, d_b, device=v.device))[0]
            rotated_bundles.append(v @ R.T)
        z_rotated = model.latent_layers[0].cat_bundles(rotated_bundles)
        y_rotated = model.decode(z_rotated)

        gauge_violation = torch.norm(y_rotated - y_original).item()
        diagnostics['node_67_gauge_invariance'] = gauge_violation

        # L1 mixing strength
        l1_strength = model.regularization_loss().item()
        diagnostics['l1_mixing_strength'] = l1_strength

        # Equivariance violation
        equiv_violation = model.equivariance_violation(z).item()
        diagnostics['equivariance_violation'] = equiv_violation

        # Mixing gate values (per layer)
        gate_values = []
        for layer in model.latent_layers:
            gates = torch.sigmoid(layer.gate_bias).cpu().numpy()
            gate_values.append(gates)
        diagnostics['mixing_gates'] = gate_values

        # Texture zeros (count entries with |W_ij| < threshold)
        threshold = 1e-3
        n_zeros = 0
        n_total = 0
        for layer in model.latent_layers:
            for i in range(layer.n_bundles):
                for j in range(layer.n_bundles):
                    W_ij = layer.mixing_weights[i][j]
                    n_total += W_ij.numel()
                    n_zeros += (torch.abs(W_ij) < threshold).sum().item()
        diagnostics['texture_zeros_count'] = n_zeros
        diagnostics['texture_zeros_fraction'] = n_zeros / n_total

    return diagnostics
```

**Expected diagnostic trajectory:**

1. **Early training (epochs 1-20):**
   - L1 strength high, equivariance violation high
   - Mixing gates near 0.5 (uncertain)
   - Few texture zeros

2. **Mid training (epochs 20-60):**
   - L1 strength decreasing as network discovers which cross-bundle couplings matter
   - Equivariance violation converging toward $\epsilon \approx 0.22$ nat/step
   - Mixing gates polarizing (some → 0, some → 1)
   - Texture zeros emerging (sparsity pattern)

3. **Late training (epochs 60-100):**
   - L1 strength stabilized
   - Equivariance violation at target $\epsilon_{\text{target}}$
   - Hierarchical structure visible: few strong couplings, many suppressed
   - Texture zeros frozen (learned symmetry-breaking pattern)

:::{admonition} Connection to Physics: Hierarchical Mixing
:class: info

The emergent sparsity pattern from L1 regularization mirrors **Froggatt-Nielsen texture zeros** in particle physics:

- Standard Model: CKM and PMNS matrices have hierarchical structure with many near-zero entries
- Fragile UGN: Mixing weights $W_{ij}$ develop similar hierarchies under L1
- Physical interpretation: Not all bundles couple equally—some interactions are "forbidden" by emergent symmetry

This is **not imposed**—it emerges from optimization under L1 penalty. The value $\epsilon \approx 0.22$ nat/step is proposed as an empirical target for investigation—analogous to the Cabibbo angle ($\sin \theta_C \approx 0.225$ in the CKM matrix)—but requires experimental validation to determine if it emerges naturally from L1 optimization.
:::



(sec-connections-outlook)=
## Connections and Outlook

:::{div} feynman-prose
And there you have it.

We started this chapter with a question: how do you take all those beautiful primitives—spectral linear layers, norm-gated activations, isotropic blocks, BAOAB integrators, Wilson lines—and assemble them into a complete, working agent? An agent that takes observations at one end and produces actions at the other, while respecting all the geometric structure we have derived.

The answer is the Universal Geometric Network, and I want to make sure you understand what we have achieved.

We started with what looked like an impossible demand. Universal approximation—the ability to learn any function—seemed to conflict with geometric constraints. Symmetry says "the network must behave the same way under rotations." Universality says "the network must be able to learn direction-dependent behavior." These requirements appear contradictory.

The resolution came from asking: *where* do constraints belong? And the answer is beautiful in its simplicity: **constraints belong in the middle, freedom at the boundaries**. The encoder is free—it can map any observation to any latent state. The decoder is free—it can map any latent state to any action. But the latent dynamics in the middle? Those respect geometry, softly, through L1 regularization that discovers the minimal symmetry breaking required by the task.

This is not a hack or a compromise. It is how gauge theories work in physics. You pick a gauge (the encoder's job). You write gauge-covariant dynamics (the latent layers' job). You compute gauge-invariant observables (the decoder's job). The gauge choice is arbitrary—that is the freedom. The dynamics are constrained—that is the geometry. And the observables come out independent of the arbitrary choice—that is consistency.

The three levels of structure we synthesized are:

1. **Micro-architecture** (Section 04): spectral linear layers, norm-gated activations, isotropic blocks—the building blocks that respect gauge structure at the level of individual operations.

2. **Dynamics** (Section 05): the Boris-BAOAB integrator, covariant cross-attention, Wilson lines—the rules for how latent states evolve over time while staying on the WFR manifold.

3. **Macro-architecture** (this chapter): the three-stage encoder-dynamics-decoder pipeline with soft equivariance—the overall shape that achieves universal approximation while preserving geometric consistency.

The result is a complete architecture. It is universal (we proved it). It is geometrically consistent (we proved that too). And it is practical—you can build it in PyTorch and train it on real tasks. The next step is empirical validation: does the L1 regularization actually discover meaningful texture zeros? Do the emergent coupling patterns tell us something about the structure of the task? That is where theory meets experiment.
:::

### Relationship to Sections 04 and 05

The Universal Geometric Network is the **synthesis** of primitives and dynamics:

| Section 04: Geometric Micro-Architecture | Section 06 (this section): End-to-End Architecture |
|-----------------------------------------|--------------------------------------------------|
| {prf:ref}`def-spectral-linear` SpectralLinear | Used in encoder/decoder for $U(1)_Y$ capacity bound |
| {prf:ref}`def-norm-gated-activation` NormGatedGELU | Replaced by softplus + norm-gating in soft-equivariant layer |
| {prf:ref}`def-isotropic-block` IsotropicBlock | Equivariant pathway $D^{\text{equiv}}$ implements this |
| {prf:ref}`thm-spectral-preserves-hypercharge` Spectral preserves light cone | Theorem {prf:ref}`thm-ugn-geometric-consistency` (1) capacity bound |

| Section 05: Covariant Cross-Attention | Section 06 (this section): End-to-End Architecture |
|---------------------------------------|--------------------------------------------------|
| {prf:ref}`def-wilson-line` Wilson line | Not explicitly used in UGN (implicit in gauge freedom), but available for explicit world models |
| {prf:ref}`def-covariant-qkv-projections` Covariant Q/K/V | Alternative to soft-equivariant layers for attention-based dynamics |
| {prf:ref}`thm-metric-temperature-correspondence` Metric = inverse temperature | Soft equivariance analog: L1 strength $\lambda_{\text{L1}}$ controls "temperature" of symmetry breaking |
| {prf:ref}`def-baoab-attention-heads` BAOAB integrator | Soft-equivariant layer implements one BAOAB step (Section {ref}`sec-universal-geometric-network` Connection to BAOAB) |

**Usage table:**

| Primitive (Section 04) | Role in UGN (Section 06) | Layer |
|--------------------|----------------------|-------|
| SpectralLinear | Encoder input layer | `self.encoder[0]` |
| SpectralLinear | Encoder hidden layer | `self.encoder[2]` |
| SpectralLinear | Decoder hidden layer | `self.decoder[0]` |
| SpectralLinear | Decoder output layer | `self.decoder[4]` |
| Norm MLP | Equivariant pathway | `SoftEquivariantLayer.norm_mlp` |
| Mixing weights | Symmetry-breaking pathway | `SoftEquivariantLayer.mixing_weights` |
| Norm-gating | Activation | `SoftEquivariantLayer` forward, final lines |

| Dynamics (Section 05) | Implementation in UGN | Component |
|-------------------|----------------------|-----------|
| B-step (gradient) | Norm MLP → scales | Equivariant pathway |
| A-step (Lorentz force) | Mixing pathway $W_{ij} v_j$ | Symmetry-breaking term |
| A-step (geodesic correction) | Norm-gating $v / \|v\| \cdot g$ | Projection to manifold |
| O-step (thermostat) | Implicit (training noise) or explicit injection | N/A |

**Integration workflow:**

1. **Build primitives (Section 04):** Define SpectralLinear, NormGatedGELU, IsotropicBlock with gauge symmetry
2. **Compose dynamics (Section 05):** Stack primitives into BAOAB integrator with covariant cross-attention
3. **Wrap in UGN (Section 06):** Add unconstrained encoder/decoder, apply L1 regularization, train end-to-end

The encoder and decoder provide **gauge freedom**—they absorb the ambiguity in choice of latent coordinates. The latent dynamics respect **geometric structure**—they evolve on the WFR manifold with soft equivariance. Together, they achieve **universal approximation** (Theorem {prf:ref}`thm-ugn-universal-approximation`) while maintaining **geometric consistency** (Theorem {prf:ref}`thm-ugn-geometric-consistency`).

### Literature Positioning

We now position the Universal Geometric Network within the broader landscape of geometric deep learning, equivariant architectures, and gauge-covariant methods.

**Table: Closest Related Work**

| Work | Group | Equivariance Type | Our Difference |
|------|-------|-------------------|----------------|
| **Geometric Deep Learning** (Bronstein et al., 2021) {cite}`bronstein2021geometric` | Varies by domain (CNNs: $SE(2)$, GNNs: $S_n$) | Spatial symmetries | We use **internal gauge group** $SU(N_f) \times SU(2) \times U(1)$, not spatial |
| **Group Equivariant CNNs** (Cohen & Welling, 2016) {cite}`cohen2016group` | Discrete rotation/reflection groups | Global transformation | We use **per-bundle** $\prod_i SO(d_b)_i$ (product group, more restrictive) |
| **SE(3)-Transformers** (Fuchs et al., 2020) {cite}`fuchs2020se3` | $SE(3)$ (rotations + translations) | Global $SE(3)$ | We use **product** of bundle rotations + **soft** equivariance |
| **Gauge Equivariant CNNs** (Cohen & Welling, 2019) {cite}`cohen2019gauge` | $SO(2)$ or $SO(3)$ on tangent spaces | **Spatial gauge** (manifold coordinates) | We use **internal gauge** (feature bundles, like particle physics) |
| **Scalars are Universal** (Villar et al., 2021) {cite}`villar2021scalars` | $O(d)$ invariance via pairwise distances | Invariant, not equivariant | We enforce **equivariance** (outputs transform) + **soft** breaking |
| **Gauge Equivariant Mesh CNNs** (de Haan et al., 2021) {cite}`dehaan2021gauge` | Gauge group on mesh surface | Spatial gauge (local frames on geometry) | Our gauge is **internal** (latent space), not tied to embedding manifold |

**Key novelties of the UGN:**

1. **Physics-derived gauge group:** $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$ emerges from first principles (multi-agent consistency, capacity constraints) rather than being chosen for convenience

2. **Product structure:** Per-bundle equivariance $\prod_{i=1}^{n_b} SO(d_b)_i$ is more restrictive than global $SO(d_b)$ (used in molecular ML). This is a *feature*: it prevents spurious cross-bundle mixing unless justified by task gradients

3. **Soft equivariance via L1:** We don't enforce exact equivariance (which limits expressiveness, Theorem {prf:ref}`thm-norm-networks-not-universal`). Instead, L1 regularization creates an inductive bias, with emergent texture zeros (Proposition {prf:ref}`prop-emergent-gauge-structure`)

4. **Encoder/decoder unconstrained:** Gauge freedom allows the encoder to choose coordinates and the decoder to interpret them. This is conceptually similar to gauge fixing in physics (choosing a gauge like Lorenz or Coulomb gauge for convenience, knowing physics is gauge-invariant)

5. **Direct sum over tensor product:** We sacrifice some physical fidelity (cross-gauge entanglement) for computational tractability. This is an *architectural approximation*, justified by bounded rationality (agents must be implementable)

**Comparison to Villar et al. (2021):**

Villar et al. prove that $O(d)$-*invariant* functions of point clouds can be written as functions of pairwise distances. This is Level 2 (Gram matrix) in our taxonomy. But:

- **Invariance ≠ Equivariance:** Their functions output scalars (invariant). We need vector-valued outputs that transform equivariantly: $f(Rz) = Rf(z)$.
- **Global vs Product:** They use global $O(d)$. We use $\prod_i SO(d_b)_i$ (independent per-bundle rotations), which is more restrictive. Only norms $\{\|v_i\|\}$ are invariant, not inner products $\langle v_i, v_j \rangle$.
- **Soft breaking:** We allow $\epsilon$-violations via L1, enabling universal approximation.

**Comparison to Gauge Equivariant CNNs:**

Cohen & Welling (2019) use "gauge equivariance" for **spatial gauge** symmetry: CNNs on manifolds that are independent of local coordinate frames (tangent space orientations). This is different from **internal gauge** symmetry in particle physics. In our setting:

- **Spatial gauge:** Ensures a CNN on a sphere gives the same answer regardless of which local chart you use to parameterize the surface. The gauge group is $SO(2)$ or $SO(3)$ acting on tangent spaces.
- **Internal gauge (ours):** Ensures the latent dynamics give the same answer regardless of which basis you use for feature bundles. The gauge group is $SU(N_f) \times SU(2) \times U(1)$ acting on internal quantum numbers (color, isospin, hypercharge), not on spacetime.

Both are called "gauge equivariance," but they refer to different symmetries. Spatial gauge is about coordinates on a manifold. Internal gauge is about the structure of the state space itself.

### Open Questions and Extensions

We conclude with directions for future work.

:::{admonition} Open Research Questions
:class: note

**1. Empirical validation of emergent texture zeros:**

Proposition {prf:ref}`prop-emergent-gauge-structure` predicts that L1 regularization will discover sparsity patterns in $W^{\text{mix}}$ without supervision. Does this happen in practice? What $\lambda_{\text{L1}}$ values are needed? How do the learned texture zeros compare to the CKM/PMNS matrices in the Standard Model?

**Experiment:** Train UGN on tasks with known equivariant structure (e.g., molecular dynamics, robotic control with rotational symmetry). Measure:
- Sparsity ratio: $\frac{\#(|W_{ij}| < \epsilon)}{\#(\text{total couplings})}$
- Hierarchical ratio: $\max |W_{ij}| / \min |W_{ij}|$ (should be $\gg 1$)
- Correlation with task symmetry: Do symmetric tasks → sparser $W$?

**2. Learned symmetry breaking parameter $\epsilon$:**

In the Standard Model, electroweak symmetry breaking occurs at energy scale $v \approx 246$ GeV. In our framework, the L1 coefficient $\lambda_{\text{L1}}$ plays an analogous role (controlling how much symmetry breaking is allowed). Can we *learn* $\lambda_{\text{L1}}$ from data via bilevel optimization or meta-learning?

**Approach:** Treat $\lambda_{\text{L1}}$ as a hyperparameter optimized on a validation set, or use a learned schedule $\lambda_{\text{L1}}(t)$ that adapts during training.

**3. Tensor product architectures:**

When is the tensor product $V_C \otimes V_L \otimes V_Y$ worth the computational cost? For low-rank structure (Frobenius norm $\ll$ full rank), factored tensor layers (Section {ref}`sec-architectural-choices`, Definition {prf:ref}`def-tensor-product-representation` remark on factorization) can be efficient. Can we design architectures that *adaptively* choose between direct sum (cheap, separable) and tensor product (expensive, entangled) based on task requirements?

**Idea:** Hybrid architecture with gating:

$$
\mathcal{Z} = \alpha \cdot (V_C \oplus V_L \oplus V_Y) + (1-\alpha) \cdot (V_C \otimes V_L \otimes V_Y)
$$
where $\alpha \in [0, 1]$ is learned. High $\alpha$ → direct sum (tractable), low $\alpha$ → tensor product (expressive).

**Note:** This requires embedding both representations into a common ambient space or using a learned projection to make the direct sum and tensor product compatible for interpolation.

**4. Anomaly cancellation as architectural constraint:**

In the Standard Model, the specific representations of quarks and leptons are required for **anomaly cancellation**—the absence of quantum inconsistencies (triangle diagrams with gauge currents). Do analogous consistency conditions arise in multi-agent settings? Can we derive architectural constraints from requiring anomaly-free gauge theories?

**Reference:** Section 8.1 ({ref}`sec-symplectic-multi-agent-field-theory`) derives the gauge group but does not address anomalies. Extension to interacting agents (multi-agent RL, game theory) might require anomaly cancellation for consistent Nash equilibria or correlated equilibria.

**5. Integration with symbolic reasoning:**

The bundle structure $\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i$ provides *interpretability*: each bundle can be associated with a semantic concept (e.g., object attributes, relational predicates, temporal modes). Can we map bundles to symbolic logic (first-order predicates, probabilistic programs) for hybrid neuro-symbolic architectures?

**Approach:** Use bundle norms $\{\|v_i\|\}$ as fuzzy truth values for predicates, and cross-bundle mixing as logical connectives. The L1-discovered texture zeros correspond to logical independence (disjoint predicate supports).

**6. Multi-scale hierarchy:**

Particle physics has a hierarchy of scales (QCD at $\Lambda_{\text{QCD}} \sim 200$ MeV, electroweak at $v \sim 246$ GeV, possibly GUT at $10^{16}$ GeV). Can we build hierarchical UGNs with bundles at multiple scales, with different equivariance constraints at each level?

**Architecture:**
- **Low-level bundles** ($\mathcal{Z}_{\text{low}}$): Strict equivariance, fine-grained features
- **Mid-level bundles** ($\mathcal{Z}_{\text{mid}}$): Soft equivariance, compositional concepts
- **High-level bundles** ($\mathcal{Z}_{\text{high}}$): Unconstrained, abstract reasoning

with learned maps between levels.
:::

### Closing Remarks

The Universal Geometric Network synthesizes three years of work on bounded rationality, gauge theory, and geometric deep learning into a single, coherent architecture. It achieves what seemed impossible at the outset: **universal approximation** (the network can represent any function) while respecting **geometric structure** (the latent space has curvature, gauge symmetry, capacity bounds derived from first principles).

The key insight—that equivariance belongs in the latent dynamics, not at the boundaries—resolves the tension between expressiveness and constraints. The encoder and decoder are unconstrained universal approximators. The latent dynamics are softly equivariant, with L1 regularization discovering the minimal symmetry breaking needed for the task.

This is not just a clever architectural trick. It reflects a deep principle: **gauge freedom**. In physics, gauge transformations are changes of description that don't affect observables. In neural architecture, the latent space representation is a choice of "mental coordinates," and gauge transformations are re-coordinatizations. The encoder chooses a gauge (implicitly, by learning), the dynamics respect gauge covariance (ensuring predictions are consistent), and the decoder extracts gauge-invariant observables (outputs don't depend on the choice of gauge).

The Universal Geometric Network is the **default architecture** for bounded agents in the Fragile framework. It integrates seamlessly with:

- {ref}`sec-geometric-micro-architecture` (geometric primitives) — uses SpectralLinear, isotropic blocks, norm-gating
- {ref}`sec-covariant-cross-attention-architecture` (geodesic dynamics) — soft-equivariant layers implement BAOAB steps
- {ref}`sec-symplectic-multi-agent-field-theory` (gauge theory) — respects $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$ structure
- {ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits` (metric law) — spectral normalization enforces capacity bounds
- {ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces` (WFR geometry) — latent space is the WFR manifold

And it makes testable predictions: emergent texture zeros, hierarchical mixing patterns, learned symmetry breaking. The next step is empirical validation.

