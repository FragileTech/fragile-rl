(sec-capacity-constrained-metric-law-geometry-from-interface-limits)=
# Capacity-Constrained Metric Law: Geometry from Interface Limits

## TLDR

- Finite boundary bandwidth implies a hard **capacity constraint**: the agent cannot stably maintain bulk information it
  cannot ground at the interface.
- Enforcing this constraint induces a **metric law**: curvature adapts as a regulator when representation approaches
  capacity saturation (a geometric form of “information bottleneck”).
- The law is not analogy-first: it is derived from information/variational structure and yields testable diagnostics.
- The practical output is a **consistency defect** and a runtime diagnostic for when the agent is claiming more
  information than its boundary can support.
- This chapter is the geometric backbone for later results: WFR belief geometry, holographic generation, and the causal
  information bound.

## Roadmap

1. State the capacity constraint and why it must hold for bounded agents.
2. Derive the metric/curvature response as a variational law.
3. Define diagnostics and implementation-facing consistency checks.

:::{div} feynman-prose
Here's a question that sounds almost silly at first: *Why should space be curved?*

In physics, Einstein gave us an answer: space curves because stuff is in it. Mass and energy bend spacetime, and that bending is what we call gravity. The geometry isn't handed down from on high---it emerges from the presence of matter.

Now, what's remarkable is that something very similar happens inside an agent. The agent has an internal "space"---its latent representation, the manifold $\mathcal{Z}$ where it organizes its beliefs about the world. And we're going to show that this space *must* be curved, and that the curvature isn't arbitrary. It's determined by a very specific physical constraint: **how much information can flow through the agent's interface with the world**.

Think about it. The agent only sees the world through a finite-bandwidth channel---its sensors. It can only act through finite-capacity motors. Everything the agent knows about the universe has to squeeze through this narrow boundary. And here's the key insight: if the agent tries to maintain more internal structure than its boundary can support, something has to give. The geometry has to respond.

This is the capacity-constrained metric law. It's not just a nice analogy to Einstein's equations---it's a structural necessity for any bounded agent.
:::

(rb-info-bottleneck-geometry)=
:::{admonition} Researcher Bridge: Information Bottleneck Becomes Geometry
:class: info
When you push a model to the edge of representational capacity, the geometry must adapt. This is the rigorous version of information bottleneck regularization: capacity limits induce curvature that slows updates in overloaded regions.
:::

{ref}`Section 9.10 <sec-differential-geometry-view-curvature-as-conditioning>` used a "gravity" analogy to motivate curvature as a regulator. This section removes the analogy: the curvature law is derived as a structural response to **information-theoretic constraints** induced by the agent's finite-bandwidth boundary (Markov blanket).

The key idea is operational: **the representational complexity of the internal state is bounded by the capacity of the interface channel.** When the agent operates near this bound (at its {prf:ref}`def-boundary-markov-blanket`), curvature appears as the geometric mechanism that prevents internal information volume from exceeding what can be grounded at the interface.

(sec-the-boundary-bulk-information-inequality)=
## The Boundary--Bulk Information Inequality

:::{div} feynman-prose
Let's start with a very simple observation that turns out to be profound.

Imagine you're an agent. You're sitting there in the world, taking in observations, making decisions. Inside your head (or your neural network, or whatever), you're building up a representation of what's going on. This representation is your internal state $Z$.

Now here's the thing: *everything you know has to come through your sensors*. There's no other way for information to get in. You can't just magically know things about the world---you have to see them, hear them, feel them. Every bit of information in your internal representation had to squeeze through your sensory boundary at some point.

This seems obvious, but it has a sharp mathematical consequence. If your boundary channel has capacity $C_\partial$ (measured in nats per unit time, say), then your internal representation can't contain more than $C_\partial$ worth of grounded information. You might have more *stuff* in there---random noise, hallucinations, unfounded beliefs---but you can't have more *information about the world* than what came through the channel.

This is the data-processing inequality, and it's one of the deepest results in information theory. You cannot create information by processing. Whatever comes out of a channel can't have more mutual information with the source than what went in.

For our purposes: $I_{\text{bulk}} \le C_\partial$. The internal information is bounded by the boundary capacity.
:::

:::{prf:definition} DPI / boundary-capacity constraint
:label: def-dpi-boundary-capacity-constraint

Consider the boundary stream $(X_t)_{t\ge 0}$ and the induced internal state process $(Z_t)_{t\ge 0}$ produced by the shutter (Definition {prf:ref}`def-bounded-rationality-controller`). Because all internal state is computed from boundary influx and internal memory, any information in the bulk must be mediated by a finite-capacity channel. Operationally, the data-processing constraint is:

$$
I_{\text{bulk}} \;\le\; C_{\partial},

$$
where $C_{\partial}$ is the effective information capacity of the boundary channel and $I_{\text{bulk}}$ is the amount of information the agent can stably maintain in $\mathcal{Z}$ without violating Causal Enclosure (no internal source term $\sigma$; Definition {prf:ref}`def-source-residual`).
Units: $[I_{\text{bulk}}]=[C_{\partial}]=\mathrm{nat}$.

:::

:::{div} feynman-prose
Now, what do we mean by "information in the bulk"? This is where things get interesting, because we need to be careful about what we're measuring.

When you have a probability distribution $\rho(z)$ over your latent space, there's a natural measure of how much information it carries: the differential entropy. But here's the subtlety---the entropy depends on what volume element you use. If you change your coordinates, the entropy changes.

This is where the metric $G$ comes in. The metric tells you how to measure volume in your latent space. And once you have a proper volume element, you can define information density in a coordinate-invariant way.

The formula might look a bit intimidating, but the idea is simple: we're counting how many nats of information the agent is carrying at each location, and we're doing it in a way that respects the geometry.
:::

:::{prf:definition} Information density and bulk information volume
:label: def-information-density-and-bulk-information-volume

Let $\rho(z,s)$ denote the probability density of the agent's belief state at position $z \in \mathcal{Z}$ and computation time $s$, **defined with respect to the Riemannian volume measure** $d\mu_G = \sqrt{|G|}\,dz^n$. The **information density** $\rho_I(z,s)\ge 0$ is defined as the local Shannon entropy density:

$$
\rho_I(z,s) := -\rho(z,s) \log \rho(z,s),

$$
with units of nats per unit Riemannian volume ($n=\dim\mathcal{Z}$).

*Remark.* Integrating $\rho_I$ over $\mathcal{Z}$ with respect to $d\mu_G$ yields the coordinate-invariant differential entropy $h[\rho] = -\int \rho \log \rho \, d\mu_G$. By defining $\rho$ as a density with respect to $d\mu_G$ (rather than Lebesgue measure $dz$), the entropy is automatically invariant under coordinate changes---no explicit metric correction term is needed.

:::

:::{note}
:class: feynman-added
The second term $\frac{1}{2}\rho \log\det G$ might seem like a technicality, but it's doing important work. Think of it this way: in a region where the metric determinant is large, the "volume is stretched"---a small coordinate box actually represents a lot of space. The information you're storing in that region is therefore worth more. This correction ensures we're measuring information in a geometrically honest way.
:::

:::{prf:definition} Bulk Information Volume
:label: def-a-bulk-information-volume

Define the bulk information volume over a region $\Omega\subseteq\mathcal{Z}$ by

$$
I_{\text{bulk}}(\Omega) := \int_{\Omega} \rho_I(z,s)\, d\mu_G.

$$
When $\Omega=\mathcal{Z}$ we write $I_{\text{bulk}}:=I_{\text{bulk}}(\mathcal{Z})$. This is conceptually distinct from the probability-mass balance in {ref}`Section 2.11 <sec-variance-value-duality-and-information-conservation>`; here the integral measures grounded structure in nats.

:::

:::{div} feynman-prose
Now here's the really interesting part: the boundary. The agent's interface with the world has a certain "area"---not physical area necessarily, but informational area. Think of it as the total number of independent channels through which information can flow.

In many physical systems, there's a remarkable phenomenon called an **area law**: the amount of information a region can hold is proportional not to its volume, but to the area of its boundary. This shows up in black hole thermodynamics, where the entropy is proportional to the horizon area. It shows up in quantum field theory, where entanglement entropy scales with boundary area. And it shows up here, in bounded agents.

Why? Because information has to get in through the boundary. If your boundary has area $A$, and each unit of area can transmit $1/\eta_\ell$ nats (where $\eta_\ell$ depends on your resolution scale), then your total capacity is $A/\eta_\ell$. It doesn't matter how big the interior is---you're bottlenecked at the boundary.

This is the deep reason why capacity constraints lead to geometry. The boundary area, measured in the metric $G$, determines how much information you can hold. If you try to hold too much, the metric has to adjust.
:::

:::{prf:definition} Boundary capacity: area law at finite resolution
:label: def-boundary-capacity-area-law-at-finite-resolution

Let $dA_G$ be the induced $(n-1)$-dimensional area form on $\partial\mathcal{Z}$. If the boundary interface has a minimal resolvable scale $\ell>0$ (pixel/token floor), then an operational capacity bound is an area law:

$$
C_{\partial}(\partial\mathcal{Z})
:=
\frac{1}{\eta_\ell}\oint_{\partial\mathcal{Z}} dA_G,

$$
where $\eta_\ell$ is the effective boundary area-per-nat at resolution $\ell$ (a resolution-dependent constant set by the interface).
Units: $[\eta_\ell]=[dA_G]/\mathrm{nat}$ and $[\ell]$ is the chosen boundary resolution length scale.

*Remark (discrete macro specialization).* For the split shutter, the most conservative computable proxy is

$$
C_{\partial}\ \approx\ \mathbb{E}[I(X_t;K_t)]\ \le\ \log|\mathcal{K}|,

$$
which is exactly Node 13 (BoundaryCheck) and Theorem {prf:ref}`thm-information-stability-window-operational`'s grounding condition.

:::

:::{admonition} Example: The Pixel Budget
:class: feynman-added example

Let's make this concrete. Suppose your agent sees the world through a 64x64 grayscale camera, and each pixel can take 256 values. What's the maximum information that can flow through this interface in one frame?

Naively, you might say $64 \times 64 \times \log(256) = 64 \times 64 \times 8 \approx 32,000$ bits. But that's almost never achieved in practice. Real images have strong correlations---neighboring pixels are usually similar. The *effective* information rate is much lower.

This is the $\eta_\ell$ factor: it accounts for the redundancy in your sensory channel. A highly compressed representation might achieve near the theoretical limit; a raw pixel stream wastes most of its bandwidth on predictable correlations.

The boundary capacity $C_\partial$ is what survives after all this redundancy is squeezed out. It's the *useful* information that actually constrains your internal representation.
:::

(sec-main-result)=
## Main Result (Capacity-Saturated Metric Law)

:::{div} feynman-prose
Alright, now we come to the main event. We've established that there's a fundamental inequality: the information in the bulk can't exceed the capacity of the boundary. What happens when the agent pushes against this limit?

The answer is beautiful: **the geometry has to change**. The metric $G$ can't stay fixed---it has to respond to the information load. And the way it responds is governed by an equation that looks remarkably like Einstein's field equations from general relativity.

Let me be clear about what's happening here. We're not doing physics. We're not saying the agent's latent space is "actually" curved spacetime. What we're saying is that the same mathematical structure---a field equation relating curvature to a source term---emerges from completely different principles. In physics, it comes from the principle of least action and the requirement that the laws be generally covariant. Here, it comes from information theory and the requirement that the bulk be grounded at the boundary.

The source term in our equation isn't the stress-energy tensor of matter. It's the "Risk Tensor"---a measure of how much the agent cares about different regions of its state space. High-value regions, regions where small changes lead to big consequences, generate curvature. The geometry responds to what matters.
:::

The detailed variational construction is recorded in {ref}`Appendix A <sec-appendix-a-full-derivations>`. The main consequence is an Euler--Lagrange identity that ties curvature of the latent geometry to a risk-induced tensor under a finite-capacity boundary.

:::{prf:theorem} Capacity-constrained metric law
:label: thm-capacity-constrained-metric-law

Under the regularity and boundary-clamping hypotheses stated in {ref}`Appendix A <sec-appendix-a-full-derivations>`, and under the soundness condition that bulk structure is boundary-grounded (no internal source term $\sigma$ on $\operatorname{int}(\mathcal{Z})$; Definition {prf:ref}`def-source-residual`), stationarity of a capacity-constrained curvature functional implies

$$
R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa\, T_{ij},

$$
where $\Lambda$ and $\kappa$ are constants and $T_{ij}$ is the **total Risk Tensor** induced by the reward field. *Units:* $\Lambda$ has the same units as curvature ($[R]\sim [z]^{-2}$), and $\kappa$ is chosen so that $\kappa\,T_{ij}$ matches those curvature units.

*Operational reading.* Curvature is the geometric mechanism that prevents the internal information volume (Definition 18.1.2a) from exceeding the boundary's information bandwidth (Definition {prf:ref}`def-a-bulk-information-volume`) while remaining grounded.

**Implementation hook.** The squared residual of this identity defines a capacity-consistency regularizer $\mathcal{L}_{\text{cap-metric}}$; see {ref}`Appendix B <sec-appendix-b-units-parameters-and-coefficients>` for the consolidated list of loss definitions and naming conventions.

:::

:::{div} feynman-prose
Let me unpack this equation piece by piece, because it's the heart of this section.

On the left side, we have:
- $R_{ij}$: the Ricci tensor. This measures how volumes change when you parallel transport them around. Positive Ricci curvature means volumes shrink; negative means they expand.
- $R$: the scalar curvature (the trace of $R_{ij}$). A single number summarizing the overall curvature at a point.
- $G_{ij}$: the metric tensor itself.
- $\Lambda$: a constant, analogous to the cosmological constant in physics. It sets a baseline curvature even when there's no "stuff" around.

On the right side:
- $\kappa$: a coupling constant. It controls how strongly the risk tensor sources curvature.
- $T_{ij}$: the Risk Tensor. This is where the agent's objectives enter the picture.

The equation says: **curvature (left side) is determined by risk (right side)**. Regions where the agent is making high-stakes decisions---where the gradient of value is steep, where small errors have big consequences---these regions will be curved. The geometry literally bends around what matters.

And here's the beautiful operational interpretation: this curvature is what prevents information overload. If you tried to maintain a high-resolution representation everywhere, you'd exceed your boundary capacity. The curvature "inflates" the important regions, giving them more effective volume, while "deflating" the unimportant ones. The total stays within budget.
:::

:::{warning}
:class: feynman-added
A common mistake is to think of this as "imposing" curvature on the latent space as a design choice. That gets the causality backwards. The curvature *emerges* from the capacity constraint. If you try to learn a flat representation while operating near capacity, you'll either exceed the information budget (and have ungrounded beliefs) or you'll implicitly learn a curved representation anyway. The field equation tells you what that curvature must be for consistency.
:::

:::{prf:definition} Extended Risk Tensor with Maxwell Stress
:label: def-extended-risk-tensor

The total Risk Tensor $T_{ij}$ decomposes into gradient and curl contributions:

$$
T_{ij} = T_{ij}^{\text{gradient}} + T_{ij}^{\text{Maxwell}},

$$
where:

1. **Gradient Stress** (from scalar potential $\Phi$):

$$
T_{ij}^{\text{gradient}} = \partial_i \Phi \, \partial_j \Phi - \frac{1}{2}G_{ij} \|\nabla\Phi\|_G^2

$$
2. **Maxwell Stress** (from {prf:ref}`def-value-curl` $\mathcal{F}$):

$$
T_{ij}^{\text{Maxwell}} = \mathcal{F}_{ik}\mathcal{F}_j^{\;k} - \frac{1}{4}G_{ij}\mathcal{F}^{kl}\mathcal{F}_{kl}

$$
*Units:* $[T_{ij}] = \mathrm{nat}^2/[z]^2$.

**Conservative Limit:** When $\mathcal{F} = 0$ (Definition {prf:ref}`def-conservative-reward-field`), the Maxwell term vanishes and we recover the standard gradient-only risk tensor.

**Non-Conservative Case:** When $\mathcal{F} \neq 0$, the Maxwell stress contributes additional terms to the curvature equation.

:::

:::{div} feynman-prose
The Risk Tensor has two pieces, and they have very different characters.

The first piece, $T_{ij}^{\text{gradient}}$, comes from the gradient of the value function. Where value changes rapidly---where you're on a steep slope in reward landscape---you get a large contribution. This makes intuitive sense: regions where small movements lead to big changes in value are "risky" and deserve extra geometric attention.

The second piece, $T_{ij}^{\text{Maxwell}}$, is more subtle. It comes from the *curl* of the reward field. In standard RL, we assume this is zero---rewards are conservative, and there's a well-defined scalar value function. But as we discussed in {ref}`Section 24 <sec-the-reward-field-value-forms-and-hodge-geometry>`, that's not always true. When the agent faces cyclic preferences (like Rock-Paper-Scissors) or when exploration-exploitation creates sustained orbits, the reward field has non-zero curl.

The Maxwell stress tells you how this cyclic structure contributes to curvature. It's called "Maxwell" because the formula is mathematically identical to the electromagnetic stress tensor in physics. In electromagnetism, this term describes how the presence of electromagnetic fields creates pressure and tension in spacetime. Here, it describes how cyclic value structures create geometric stress in the latent space.

For most practical purposes, you can ignore the Maxwell term---most reward functions are conservative. But when they're not, this term explains the geometric consequences.
:::

:::{admonition} Example: The Cliff Walk
:class: feynman-added example

Consider a classic "cliff walking" problem. The agent must traverse a path where one side is safe (low reward) and the other side is a cliff (large negative reward for falling off).

Near the cliff edge, the gradient of value is enormous---a small step in the wrong direction means disaster. The Risk Tensor is large here. According to our metric law, this region will have high curvature.

What does high curvature do operationally? It effectively "inflates" the danger zone. The geodesic distance between "safe" and "danger" becomes larger than the coordinate distance would suggest. When the agent is using geodesic motion (following natural gradients), it automatically slows down and takes smaller steps near the cliff. The geometry encodes caution.

This is exactly what you'd want from a bounded agent. It doesn't have infinite precision, so it needs to be more careful where precision matters. The capacity-constrained metric law makes this happen automatically.
:::

(pi-einstein-equations)=
::::{admonition} Physics Isomorphism: Einstein Field Equations
:class: note

**In Physics:** Einstein's field equations relate spacetime curvature to stress-energy: $R_{\mu\nu} - \frac{1}{2}Rg_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}$ {cite}`einstein1915field,wald1984general`.

**In Implementation:** The capacity-constrained metric law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) relates latent geometry to risk:

$$
R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa T_{ij}

$$
**Correspondence Table:**

| General Relativity | Agent (Metric Law) |
|:-------------------|:-------------------|
| Spacetime metric $g_{\mu\nu}$ | Latent metric $G_{ij}$ |
| Ricci tensor $R_{\mu\nu}$ | Ricci tensor $R_{ij}$ (of $G$) |
| Cosmological constant $\Lambda$ | Baseline curvature $\Lambda$ |
| Stress-energy $T_{\mu\nu}$ | Risk tensor $T_{ij}$ |
| Gravitational coupling $8\pi G$ | Capacity coupling $\kappa$ |
| Schwarzschild horizon | Saturation horizon (Lemma {prf:ref}`lem-metric-divergence-at-saturation`) |

**Loss Function:** $\mathcal{L}_{\text{EFE}} := \|R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} - \kappa T_{ij}\|_F^2$.
::::

:::{div} feynman-prose
I want to be careful about what this isomorphism means and what it doesn't mean.

It *does* mean that the same mathematical structure appears in both contexts. The Einstein tensor (left side of the equation) and its relationship to a source tensor (right side) arise from very general variational principles. Whenever you're optimizing a functional that depends on a metric and some matter fields, subject to diffeomorphism invariance (coordinate freedom), you tend to get equations of this form. It's a deep theorem in differential geometry.

It *doesn't* mean that the agent's latent space is "really" a spacetime, or that there's actual gravity involved. The physics is completely different. In GR, the metric is a property of the arena in which events occur. Here, the metric is a property of the agent's *representation*---it's a learned structure that organizes information efficiently.

But the structural similarity is useful. We can borrow intuitions: "mass curves spacetime" becomes "risk curves latent space." We can borrow techniques: the numerical methods for solving Einstein's equations might be adapted for learning optimal metrics. And we can borrow warnings: the singularities that plague GR (black holes, the Big Bang) have analogues here as "capacity saturation horizons" where the representation breaks down.
:::

(sec-diagnostic-node-capacity-saturation)=
## Diagnostic Node: Capacity Saturation

:::{div} feynman-prose
How do you know when you're hitting the capacity limit? That's what this diagnostic monitors.

The idea is simple: compute the ratio of bulk information to boundary capacity. If this ratio is close to 1, you're operating near the edge. If it exceeds 1, something has gone wrong---you're claiming to know more than your sensors could possibly have told you.

When the ratio is small, you have headroom. The agent is underutilizing its interface, and there's room to build richer internal representations. But when the ratio approaches 1, the geometry must start responding. The curvature corrections become significant.

Think of it like the pressure in a balloon. When the balloon is half-full, the walls are barely stressed. As you inflate it toward capacity, the tension in the walls increases. Push past the limit, and it pops. For our agent, "popping" means having ungrounded beliefs---internal structure that isn't supported by the boundary data. That's a failure mode we want to detect and avoid.
:::

| #  | Name                    | Measures                        | Trigger                                         |
|----|-------------------------|---------------------------------|-------------------------------------------------|
| 40 | CapacitySaturationCheck | Bulk-boundary information ratio | $I_{\text{bulk}} / C_{\partial} > 1 - \epsilon$ |

:::{prf:definition} Capacity saturation diagnostic
:label: def-capacity-saturation-diagnostic

Compute the capacity saturation ratio:

$$
\nu_{\text{cap}}(s) := \frac{I_{\text{bulk}}(s)}{C_{\partial}},

$$
where $I_{\text{bulk}}(s) = \int_{\mathcal{Z}} \rho_I(z,s)\, d\mu_G$ per Definition 18.1.2a.

*Interpretation:*
- $\nu_{\text{cap}} \ll 1$: Under-utilized capacity; the agent may be compressing excessively (lossy representation).
- $\nu_{\text{cap}} \approx 1$: Operating at capacity limit; geometry must regulate to prevent overflow.
- $\nu_{\text{cap}} > 1$: **Violation** of the DPI constraint (Definition {prf:ref}`def-dpi-boundary-capacity-constraint`); indicates ungrounded structure.

*Cross-reference:* When $\nu_{\text{cap}} > 1$, the curvature correction (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) is insufficient. This triggers geometric reflow---the metric $G$ must increase $|G|$ (expand volume) to bring $I_{\text{bulk}}$ back within bounds.

:::

:::{admonition} What "Geometric Reflow" Looks Like
:class: feynman-added tip

When the capacity saturation diagnostic triggers, what actually happens? The agent needs to "expand" its latent space to accommodate the information load without exceeding the boundary capacity.

Mathematically, this means increasing the metric determinant $|G|$. Larger $|G|$ means larger volumes, and larger volumes mean the same probability distribution is "spread out" over more space, carrying less information density.

In practice, this might manifest as:
1. **Slower learning rates** in saturated regions (the natural gradient gets smaller)
2. **Increased uncertainty** in beliefs (the distribution becomes more diffuse)
3. **Coarser representations** (fewer effective degrees of freedom distinguish nearby states)

All of these are adaptive responses to hitting the information wall. The agent can't magically process more information than its sensors provide, so it has to work with what it's got.
:::

::::{admonition} Connection to RL #25: Information Bottleneck as Degenerate Capacity-Constrained Metric
:class: note
:name: conn-rl-25
**The General Law (Fragile Agent):**
The latent metric obeys a **Capacity-Constrained Consistency Law** (Theorem {prf:ref}`thm-capacity-constrained-metric-law`):

$$
R_{ij} - \frac{1}{2}R\, G_{ij} + \Lambda G_{ij} = \kappa\, T_{ij}

$$
where $R_{ij}$ is Ricci curvature and $T_{ij}$ is the Risk Tensor. The constraint is the **DPI inequality**: $I_{\text{bulk}} \le C_\partial \le \log|\mathcal{K}|$.

**The Degenerate Limit:**
Remove geometric structure ($G \to I$, $R_{ij} \to 0$). Replace the area law with a scalar rate constraint $\beta$.

**The Special Case (Standard RL):**

$$
\max_\theta I(Z; Y) - \beta I(Z; X)

$$
This recovers the **Information Bottleneck** {cite}`tishby2015ib` and **Variational Information Bottleneck (VIB)** {cite}`alemi2016vib`.

**What the generalization offers:**
- **Geometric response**: Curvature *emerges* from capacity constraints---it's not imposed by hand
- **Area law**: Boundary capacity scales with interface area $C_\partial \sim \text{Area}(\partial\mathcal{Z})$, not arbitrary $\beta$
- **Grounded structure**: Bulk information must be mediated by finite-bandwidth boundary (DPI)
- **Diagnostic saturation**: CapacitySaturationCheck (Node 40) monitors $\nu_{\text{cap}} = I_{\text{bulk}}/C_\partial$ at runtime
::::

:::{div} feynman-prose
Let me close this section by emphasizing what we've accomplished.

We started with a simple observation: bounded agents can only know what their sensors tell them. This is the data-processing inequality, and it's inviolable.

We then asked: what happens when the agent tries to represent more than its boundary can support? The answer is that the geometry must respond. The metric on the latent space has to curve in such a way that the total information stays within budget.

The equation governing this response looks exactly like Einstein's field equations, but with risk replacing energy as the source. This isn't a coincidence or a metaphor---it's a structural consequence of variational principles and the requirement that the representation be self-consistent.

And finally, we have a diagnostic: if the capacity saturation ratio exceeds 1, something has gone wrong. The agent is claiming to know things it couldn't possibly have learned from its sensors. That's a bug, not a feature, and this framework lets you detect it.

The capacity-constrained metric law is the geometric version of "you can't get something for nothing." Information has to come from somewhere, and the geometry keeps honest books.
:::
