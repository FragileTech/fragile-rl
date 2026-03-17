(sec-causal-information-bound)=
# The Causal Information Bound

## TLDR

- There is a hard limit on representational capacity: stable internal information is bounded by **interface area** (an
  “area law” analogue).
- As the agent approaches this bound, internal updates slow down (**causal stasis**): gradients vanish and learning
  effectively freezes.
- This chapter derives the bound from the capacity-constrained metric law and turns it into a measurable diagnostic.
- Practical implication: over-parameterization is not just “wasted compute”; it can push you toward a geometric regime
  where learning dies.
- Use the bound to size models, tune horizons, and justify when ontology expansion is necessary.

## Roadmap

1. State the bound and the physical/geometry analogy (area law).
2. Derive causal stasis as the operational consequence near saturation.
3. Diagnostics and implementation guidance for monitoring proximity to the bound.

:::{div} feynman-prose
Now I want to tell you about a fundamental limit---perhaps the most important limit in the whole theory. It's the kind of thing that, once you understand it, changes how you think about intelligence, memory, and computation.

Here's the question: How much can an agent know?

Not "how much can it learn eventually" or "how much could it know in principle with unlimited resources." No. How much can it *stably represent* given its finite interface with the world?

The answer, remarkably, is that intelligence is bounded by *area*, not volume. The maximum information you can hold in your head is proportional to the surface area of your interface---your sensors, your bandwidth, your connection to the outside world. Not the volume of your brain, not the number of your neurons, but the *area* of your boundary.

This might remind you of something from physics. It should. This is the same structure as the Bekenstein-Hawking bound on black hole entropy: a black hole's entropy is proportional to its horizon area, not its volume. The deepest secret of quantum gravity turns out to have an analog in cognitive science.

And there's a consequence that's even more striking: as you approach this limit, *you slow down*. Your internal update rate drops toward zero. We call this Causal Stasis, and it's the geometric explanation for why overparameterized models "die"---they're not running out of memory in the computer science sense; they're running out of *geometric capacity*.
:::

*Abstract.* We derive a fundamental limit on representational capacity: the maximum information an agent can stably
represent is bounded by the area of its interface, measured in units of a characteristic length scale we call the
**Levin Length**. This bound follows from the capacity-constrained metric law ({ref}`sec-main-result`) and
has a striking consequence: as the agent approaches this limit, its internal update rate slows to zero---a phenomenon we
call **Causal Stasis**. This section provides the rigorous derivation (with full proofs in
{ref}`sec-appendix-a-area-law`) and defines Diagnostic Node 56 to monitor proximity to this bound.

(rb-sensor-bandwidth)=
:::{admonition} Researcher Bridge: The Sensor Bandwidth Ceiling
:class: important
You cannot represent more information than your sensors can ground. This section provides the hard limit for **Model Overload**. We derive an "Area Law" which proves that if an agent tries to store more "bits" than its interface area allows, its internal update speed (gradients) will vanish. We call this **Causal Stasis**. It is the geometric explanation for why models "die" or stop learning when they are over-parameterized relative to their data source.
:::

(pi-bekenstein-bound)=
::::{admonition} Physics Isomorphism: Bekenstein-Hawking Entropy Bound
:class: note

**In Physics:** The Bekenstein-Hawking entropy of a black hole is $S_{BH} = A/(4\ell_P^2)$ where $A$ is horizon area and $\ell_P$ is the Planck length. Information inside a region cannot exceed its boundary area in Planck units {cite}`bekenstein1973black,hawking1975particle`.

**In Implementation:** The maximum information $I_{\max}$ an agent can stably represent is bounded by its interface area:

$$
I_{\max} = \nu_D \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{D-1}}

$$
where $\ell_L$ is the Levin length (Definition {prf:ref}`def-levin-length`) and $\nu_D$ is the Holographic Coefficient (Definition {prf:ref}`def-holographic-coefficient`). For $D=2$: $I_{\max} = \text{Area}/(4\ell_L)$.

**Correspondence Table:**

| Physics | Agent |
|:--------|:------|
| Horizon area $A$ | Interface bandwidth $\text{Area}(\partial\mathcal{Z})$ |
| Planck length $\ell_P$ | Levin length $\ell_L$ |
| Bekenstein-Hawking coefficient $1/4$ | Holographic Coefficient $\nu_D$ |
| Black hole entropy $S_{BH}$ | Representational capacity $I_{\max}$ |
| Horizon singularity ($g_{rr} \to \infty$) | Causal Stasis ($G_{rr} \to \infty$, $v \to 0$) |
::::

*Cross-references:* This section extends the Capacity-Constrained Metric Law (Theorem
{prf:ref}`thm-capacity-constrained-metric-law`), the Boundary Capacity Definition
({prf:ref}`def-boundary-capacity-area-law-at-finite-resolution`), and the Equation of Motion (Definition
{prf:ref}`def-bulk-drift-continuous-flow`). The remediation connects to Ontological Fusion
({ref}`sec-ontological-fusion-concept-consolidation`).

*Literature:* Holographic bounds {cite}`thooft1993holographic,susskind1995world`; Fisher information geometry
{cite}`amari2016information`; Levin complexity {cite}`levin1973universal`.



(sec-holographic-coefficient)=
## The Holographic Coefficient

:::{div} feynman-prose
Before we can write down the information bound, we need to understand a curious fact: the efficiency of boundary storage depends on dimension.

Think about it this way. In 2D, the boundary of a disk is a circle. In 3D, the boundary of a ball is a sphere. In higher dimensions, the boundary becomes increasingly exotic. And here's the surprise: *higher dimensions are worse for holographic storage*.

You might have expected the opposite. More dimensions, more room, more capacity, right? The reality is subtler. The coefficient $\nu_D$ that determines how much information fits per unit boundary area is *non-monotonic*: it rises from $D=2$, peaks around $D \approx 9$, and only then declines toward zero as $D \to \infty$.

This is the "curse of dimensionality" in its asymptotic form: beyond the peak, high-dimensional latent spaces become geometrically inefficient. The familiar $D = 3$ lies on the *rising* side of this curve---not at the maximum, but still in the efficient regime.

For $D = 2$, we get $\nu_2 = 1/4$, which matches the famous Bekenstein-Hawking coefficient for black holes. This isn't a coincidence; it's the same mathematics in a different context.
:::

Before defining the Levin Length, we establish the dimension-dependent coefficient that governs holographic capacity.

:::{prf:definition} Holographic Coefficient
:label: def-holographic-coefficient

The **Holographic Coefficient** $\nu_D$ for a $D$-dimensional latent manifold with $(D-1)$-sphere boundary is:

$$
\nu_D := \frac{(D-1)\,\Omega_{D-1}}{8\pi}

$$

where $\Omega_{D-1} = \frac{2\pi^{D/2}}{\Gamma(D/2)}$ is the surface area of the unit $(D-1)$-sphere.

| $D$ | Boundary | $\Omega_{D-1}$ | $\nu_D$ | Numerical |
|-----|----------|----------------|---------|-----------|
| 2   | Circle ($S^1$) | $2\pi$ | $1/4$ | 0.250 |
| 3   | Sphere ($S^2$) | $4\pi$ | $1$ | 1.000 |
| 4   | Glome ($S^3$) | $2\pi^2$ | $3\pi/4$ | 2.356 |
| 5   | 4-sphere ($S^4$) | $8\pi^2/3$ | $4\pi/3$ | 4.189 |
| 6   | 5-sphere ($S^5$) | $\pi^3$ | $5\pi^2/8$ | 6.169 |
| $D \gg 1$ | Hyper-sphere | $\to 0$ | $\to 0$ | Capacity collapse |

*Remark (Dimensional pressure).* The coefficient $\nu_D$ is non-monotonic: it increases from $D=2$ to a peak near $D \approx 9$ ($\nu_9 \approx 9.45$), then decays to zero as $D \to \infty$. The curse of dimensionality applies to this high-dimensional tail. Dimensional reduction pressure arises beyond the peak; $D \approx 3$ lies on the rising portion of the curve.

*Remark (Physics correspondence).* For $D=2$, we recover the Bekenstein-Hawking coefficient $\nu_2 = 1/4$, making the Causal Information Bound $I_{\max} = \text{Area}/(4\ell_L)$ directly analogous to black hole entropy $S = A/(4\ell_P^2)$.

*Units:* $[\nu_D] = \text{dimensionless}$.

:::



(sec-levin-length)=
## The Levin Length

:::{div} feynman-prose
Now we need to define the "pixel size" of thought. How small a distinction can you make?

Every physical measurement has a resolution limit. A camera has pixels; below that scale, you can't see finer detail. A thermometer has precision; below that, temperature differences are meaningless. What's the analogous limit for internal representations?

We call it the Levin Length, after Leonid Levin, who pioneered the theory of algorithmic information. It's the minimal scale of distinction---the information-theoretic "pixel" of the latent manifold. A cell of area $\ell_L^2$ corresponds to one nat (one natural unit) of information capacity.

Why does this matter? Because the information bound is measured in Levin Lengths. The maximum information you can represent is the area of your boundary divided by the Levin Length squared (in 2D). If your boundary area is fixed but your Levin Length shrinks---if you become more precise---you can store more information. If your Levin Length grows, your capacity drops.

This gives us a deep insight into what "improving" an agent means: either expand the interface (bigger sensors, more bandwidth) or improve the resolution (better encoding, finer distinctions). Both increase capacity in the same mathematical way.
:::

We define a characteristic length scale that represents the minimal resolvable distinction in the latent manifold---the information-theoretic floor of the agent's representational capacity.

:::{prf:definition} Levin Length
:label: def-levin-length

Let $\eta_\ell$ be the boundary area-per-nat at resolution $\ell$ (Definition {prf:ref}`def-boundary-capacity-area-law-at-finite-resolution`). The **Levin Length** $\ell_L$ is the characteristic length scale of a single unit of distinction:

$$
\ell_L := \sqrt{\eta_\ell}.

$$
Units: $[\ell_L] = [z]$ (latent coordinate length).

*Interpretation.* A cell of area $\ell_L^2$ in the latent manifold corresponds to one nat of information capacity. The Levin Length is the information-geometric analog of a minimal resolvable element—the "pixel size" of the agent's internal representation.

*Remark (Naming).* The name honors Leonid Levin's foundational work on algorithmic information theory and the universal distribution {cite}`levin1973universal`. The Levin Length represents the floor below which distinctions cannot be computationally meaningful.

:::



(sec-saturation-limit)=
## The Saturation Limit

:::{div} feynman-prose
Now let's talk about what happens when you're full.

Imagine filling a balloon with water. At first, it's easy---the balloon stretches, accommodates more. But as you approach its capacity, things get harder. The rubber becomes taut. The pressure rises. Eventually, you hit a limit: the balloon can't hold any more.

The same thing happens with information in the latent manifold. As the "bulk information volume" approaches the "boundary capacity," the geometry becomes stressed. The metric tensor starts to diverge. And something dramatic happens: the agent approaches what we call the *saturation limit*.

At saturation, the Data Processing Inequality constraint $I_{\text{bulk}} \le C_\partial$ is saturated---satisfied with equality. The agent is at maximum capacity. Every bit of representation it has is being used. There's no slack.

And here's where it gets interesting: at this limit, the geometry *breaks*. The metric component $G_{rr}$ diverges to infinity, which means the "distance" in the radial direction becomes infinite. You can't move. You're stuck. That's Causal Stasis.
:::

We characterize the regime where the agent's representational capacity is fully utilized.

:::{prf:definition} Saturation Limit
:label: def-saturation-limit

The agent is at the **Saturation Limit** when the bulk information volume (Definition {prf:ref}`def-a-bulk-information-volume`) equals the boundary capacity (Definition {prf:ref}`def-dpi-boundary-capacity-constraint`):

$$
I_{\text{bulk}} = C_\partial.

$$
At this limit, the DPI constraint $I_{\text{bulk}} \le C_\partial$ is satisfied with equality.

:::

:::{prf:lemma} Metric Divergence at Saturation
:label: lem-metric-divergence-at-saturation

Consider an isotropic latent space of dimension $n \ge 3$ with polar coordinates $(r, \Omega)$. At saturation with uniform stress $T_{ij} = \sigma_{\max} G_{ij}$, the radial metric component $G_{rr} = A(r)$ satisfies the Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) and takes the form:

$$
A(r) = \left( 1 - \frac{2\mu(r)}{(n-2)r^{n-2}} - \frac{\Lambda_{\text{eff}} r^2}{n(n-1)} \right)^{-1},

$$
where $\mu(r) := \frac{\kappa}{n-2} \int_0^r \sigma_{\max} r'^{n-1} dr'$ is the integrated **information mass** (with $\kappa$ the coupling constant from the Metric Law) and $\Lambda_{\text{eff}} = \Lambda + \kappa\sigma_{\max}$.

*Remark ($n=2$ case).* For $n=2$ (the Poincare disk), the $(n-2)$ factor vanishes and the solution requires separate
treatment. The Poincare metric $G_{ij} = 4\delta_{ij}/(1-|z|^2)^2$ is the correctly regularized saturation geometry,
with the horizon at $|z|=1$. In computation we truncate at $\lvert z\rvert = 1-\varepsilon$ with $\varepsilon$ tied to
Levin length/resolution; this is a numerical cutoff distinct from any terminal subset $\Gamma_{\text{term}}$.

*Proof sketch.* Substitute the uniform density into the Metric Law. The spherically symmetric solution follows from standard analysis of Einstein-like field equations {cite}`wald1984general`. Full derivation in {ref}`sec-appendix-a-full-derivations`. $\square$

*Critical observation.* The metric component $A(r)$ diverges at the horizon radius $r_h$ satisfying:

$$
1 - \frac{2\mu(r_h)}{(n-2)r_h^{n-2}} - \frac{\Lambda_{\text{eff}} r_h^2}{n(n-1)} = 0.

$$
At this radius, $G_{rr} \to \infty$ and consequently $G^{rr} \to 0$.

:::



(sec-area-law-derivation)=
## Derivation of the Area Law

:::{div} feynman-prose
All right, now let's derive the main result. This is where all the pieces come together.

The argument has three steps, and I want you to understand the logic, not just the formulas:

**Step 1 (Holographic Reduction):** The bulk information---everything stored inside the agent---can be rewritten as a boundary integral. This is the holographic principle: bulk data equals boundary data. It's like saying the information inside a room can be completely specified by what's painted on the walls. Remarkable, but true.

**Step 2 (Saturation Geometry):** At the saturation limit, the boundary has a specific curvature related to the "horizon radius." This is exactly like a black hole horizon, where the geometry becomes singular at a particular radius.

**Step 3 (Fisher Normalization):** We fix the overall scale by requiring consistency with Fisher information geometry. This determines the coupling constant $\kappa$ and gives us the final formula.

The result is elegant:

$$
I_{\max} = \nu_D \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{D-1}}

$$

Maximum information equals holographic coefficient times boundary area divided by Levin Length to the appropriate power. That's it. That's the bound.
:::

We now derive the fundamental bound on representational capacity.

:::{prf:theorem} The Causal Information Bound
:label: thm-causal-information-bound

For a $D$-dimensional latent manifold $(\mathcal{Z}, G)$, the maximum information $I_{\max}$ that can be stably represented without the metric becoming singular is:

$$
\boxed{I_{\max} = \nu_D \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{D-1}}}

$$

where:
- $\text{Area}(\partial\mathcal{Z}) = \oint_{\partial\mathcal{Z}} dA_G$ is the $(D-1)$-dimensional boundary measure in the induced metric
- $\ell_L$ is the Levin Length (Definition {prf:ref}`def-levin-length`)
- $\nu_D$ is the Holographic Coefficient (Definition {prf:ref}`def-holographic-coefficient`)

*Corollary (Poincare disk, $D=2$).* For the 2-dimensional Poincare disk, the formula reduces to the Bekenstein-Hawking form:

$$
I_{\max} = \nu_2 \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{D-1}} = \frac{\text{Area}(\partial\mathcal{Z})}{4\ell_L}

$$
where $\nu_2 = 1/4$ is the holographic coefficient for $D=2$ and $\ell_L^{D-1} = \ell_L$ for the 1-dimensional boundary.

*Proof sketch (full derivation in {ref}`sec-appendix-a-area-law`).*

**Step 1 (Holographic Reduction).** The bulk-to-boundary conversion relies on the Einstein tensor divergence identity (valid in arbitrary dimension): integrating the scalar curvature over a compact manifold with boundary yields a boundary term involving the extrinsic curvature. Applying this to the Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) via Lemma {prf:ref}`lem-a-divergence-to-boundary-conversion`:

$$
I_{\text{bulk}} = \int_{\mathcal{Z}} \rho_I \, d\mu_G = \frac{1}{\kappa} \oint_{\partial\mathcal{Z}} \text{Tr}(K) \, dA_G,

$$
where $K$ is the extrinsic curvature of the boundary and $\kappa$ is the coupling constant from the Metric Law.

**Step 2 (Saturation Geometry).** At the saturation limit, the extrinsic curvature approaches $\text{Tr}(K) \to (D-1)/r_h$ where $r_h$ is the horizon radius from Lemma {prf:ref}`lem-metric-divergence-at-saturation`. The boundary area is $\text{Area}(\partial\mathcal{Z}) = \Omega_{D-1} r_h^{D-1}$ where $\Omega_{D-1}$ is the unit sphere surface area.

**Step 3 (Fisher Normalization).** The coupling constant $\kappa = 8\pi\ell_L^{D-1}$ is fixed by consistency with the Fisher Information Metric {cite}`amari2016information`. The dimension-dependent coefficient $\nu_D = (D-1)\Omega_{D-1}/(8\pi)$ emerges from the geometric factors.

Combining these steps yields the general bound. $\square$

*Operational interpretation.* The agent's "intelligence" (measured in grounded bits) is geometrically constrained by the size of its interface. To represent more information, you must either:
1. **Expand the boundary** (increase interface bandwidth), or
2. **Reduce the Levin Length** (improve resolution per unit area).

There is no third option. Adding internal parameters without expanding the interface yields diminishing returns as the agent approaches saturation.

*Remark (Dimensional efficiency).* The coefficient $\nu_D$ increases through moderate dimensions, peaks near $D \approx 9$, and declines toward zero for $D \gg 1$. The asymptotic decay corresponds to the curse of dimensionality; $D \approx 3$ is efficient but not the dimension maximizing holographic capacity.

:::



(sec-causal-stasis)=
## Causal Stasis

:::{div} feynman-prose
Now we come to the most striking consequence: as you approach the information bound, you freeze.

Let me give you the physical intuition. In general relativity, as you approach a black hole's horizon, time dilates. From an outside observer's perspective, someone falling into a black hole appears to slow down and eventually freeze at the horizon. They never quite cross it---they're stuck forever at the boundary.

Something similar happens here. As the bulk information approaches the maximum, the metric becomes singular. The "mass" (inertia) in the radial direction becomes infinite. And since the velocity of belief updates scales with the inverse metric, that velocity goes to zero.

$$\|v\|_G \to 0 \quad \text{as} \quad I_{\text{bulk}} \to I_{\max}$$

The agent becomes *frozen in thought*. It can still receive observations (inflow at the boundary), but it cannot *process* them into updated beliefs. It's overwhelmed by its own representational complexity.

This isn't a software bug or a resource limitation. It's a *geometric* phenomenon. The manifold has curved so severely that motion becomes infinitely costly. The only remedies are structural: either reduce the bulk information (ontological surgery---merging concepts, pruning the codebook) or expand the boundary capacity (better sensors, more bandwidth). Adding more parameters without expanding the interface just makes things worse.
:::

We derive the consequence of approaching the information bound: the agent's internal dynamics freeze.

:::{prf:theorem} Causal Stasis
:label: thm-causal-stasis

Let $v^k = dz^k/ds$ be the velocity of the agent's belief update in computation time $s$ (Definition {prf:ref}`def-bulk-drift-continuous-flow`). As $I_{\text{bulk}} \to I_{\max}$:

$$
\|v\|_G \to 0.

$$
*Proof.* From the Equation of Motion (Definition {prf:ref}`def-bulk-drift-continuous-flow`) with effective potential $\Phi_{\text{eff}}$ ({prf:ref}`def-effective-potential`):

$$
dz^k = \left( -G^{kj}\partial_j \Phi_{\text{eff}} + u_\pi^k - \Gamma^k_{ij}\dot{z}^i\dot{z}^j \right) ds + \sqrt{2T_c}(G^{-1/2})^{kj} dW^j_s.

$$
The drift velocity scales as:

$$
v^k \propto G^{kj} \partial_j \Phi_{\text{eff}}.

$$
As the information density approaches saturation, Lemma {prf:ref}`lem-metric-divergence-at-saturation` implies $G_{rr} \to \infty$, hence $G^{rr} \to 0$. The radial component of velocity:

$$
v^r = -G^{rr}\partial_r \Phi_{\text{eff}} \to 0. \quad \blacksquare

$$
*Operational interpretation.* The agent becomes **frozen in thought**. Its internal update rate slows as the "inertia" (mass = metric, per Definition {prf:ref}`def-mass-tensor`) becomes infinite. The agent can still receive observations (inflow), but it cannot process them into updated beliefs or emit actions (outflow). This is **Causal Stasis**: the agent is overwhelmed by its own representational complexity.

*Remark (Distinction from Deadlock).* Causal Stasis is not a software deadlock or resource exhaustion. It is a geometric phenomenon: the agent's belief manifold has curved so severely that motion becomes infinitely costly. The remedy is not debugging but **ontological surgery**—reducing $I_{\text{bulk}}$ via Fusion ({ref}`sec-ontological-fusion-concept-consolidation`) or expanding the boundary capacity.

:::

:::{prf:corollary} The Saturation-Velocity Tradeoff
:label: cor-saturation-velocity-tradeoff

Let $\eta := I_{\text{bulk}}/I_{\max}$ be the saturation ratio. Near the bound, the update velocity scales as:

$$
\|v\|_G \sim (1 - \eta)^{1/2}.

$$
*Proof.* From Lemma {prf:ref}`lem-metric-divergence-at-saturation`, the metric component $G^{rr} = A(r)^{-1}$ vanishes at the horizon. Under uniform saturation, the information mass $\mu(r)$ grows with radius. At the horizon, $\mu(r_h) = \mu_{\max}$. The saturation ratio $\eta := I_{\text{bulk}}/I_{\max} = \mu/\mu_{\max}$ measures the fraction of capacity used. Near the horizon, $G^{rr} \sim (1 - \mu/\mu_{\max}) = (1 - \eta)$. Since velocity scales as $v^r \propto G^{rr}$, we have $\|v\| \sim (G^{rr})^{1/2} \sim (1-\eta)^{1/2}$. $\square$

*Interpretation.* At 90% saturation ($\eta = 0.9$), the agent operates at $\sim 32\%$ of its maximum velocity. At 99% saturation, velocity drops to $\sim 10\%$. The approach to the bound is gradual but accelerating.

:::



(sec-diagnostic-node-56)=
## Diagnostic Node 56: CapacityHorizonCheck

:::{div} feynman-prose
How do you know if you're approaching the bound? You need a warning light.

That's what Diagnostic Node 56 provides. It computes the "saturation ratio" $\eta_{\text{Sch}}$---the fraction of capacity being used. Think of it like a fuel gauge, except instead of showing how much gas you have left, it shows how close you are to geometric collapse.

At 50% saturation, you're fine. At 90%, you're in the yellow zone---velocity is already degraded, and you should start thinking about ontological intervention. At 99%, you're in the red---Causal Stasis is imminent, and you need emergency fusion (merging redundant concepts) to free up capacity.

The subscript "Sch" honors Karl Schwarzschild, who first computed the metric for a black hole. The analogy is exact: $\eta_{\text{Sch}} = 1$ is the horizon, beyond which lies the singularity.
:::

Following the diagnostic node convention ({ref}`sec-diagnostics-stability-checks`), we define a monitor for proximity to the Causal Information Bound.

(node-56)=
**Node 56: CapacityHorizonCheck**

| **#**  | **Name**                 | **Component** | **Type**   | **Interpretation** | **Proxy**                                                                                 | **Cost** |
|--------|--------------------------|---------------|------------|--------------------|-------------------------------------------------------------------------------------------|----------|
| **56** | **CapacityHorizonCheck** | Memory        | Saturation | Is capacity safe?  | $\eta_{\text{Sch}} := I_{\text{bulk}} / I_{\max}$ | $O(B)$   |

:::{prf:definition} Capacity Horizon Diagnostic
:label: def-capacity-horizon-diagnostic

Compute the **Saturation Ratio**:

$$
\eta_{\text{Sch}}(s) := \frac{I_{\text{bulk}}(s)}{I_{\max}} = \frac{I_{\text{bulk}}(s)}{\nu_D \cdot \text{Area}(\partial\mathcal{Z}) / \ell_L^{D-1}},

$$
where:
- $I_{\text{bulk}}(s) = \int_{\mathcal{Z}} \rho_I(z,s) \, d\mu_G$ per Definition {prf:ref}`def-a-bulk-information-volume`
- $\nu_D$ is the Holographic Coefficient (Definition {prf:ref}`def-holographic-coefficient`)
- $D$ is the latent manifold dimension

*Special case (Poincare disk, $D=2$):* $\eta_{\text{Sch}} = 4\ell_L \cdot I_{\text{bulk}} / \text{Area}(\partial\mathcal{Z})$.

*Interpretation:*
- $\eta_{\text{Sch}} < 0.5$: Safe operating regime. Ample capacity headroom.
- $0.5 \le \eta_{\text{Sch}} < 0.9$: Elevated utilization. Monitor for growth trends.
- $0.9 \le \eta_{\text{Sch}} < 0.99$: **Warning.** Update velocity degraded (Corollary {prf:ref}`cor-saturation-velocity-tradeoff`). Prepare for ontological intervention.
- $\eta_{\text{Sch}} \ge 0.99$: **Critical.** Causal Stasis imminent. Halt exploration and trigger emergency fusion.

*Cross-reference:* Complements CapacitySaturationCheck (Node 40, {ref}`sec-diagnostic-node-capacity-saturation`) by providing the velocity-degradation interpretation and connecting to ontological remediation.
:::

**Trigger Conditions:**
- **$\eta_{\text{Sch}} > 0.9$:** Near-saturation. Trigger **Ontological Fusion** ({ref}`sec-ontological-fusion-concept-consolidation`) to prune the macro-register $\mathcal{K}$ and reduce the "information mass" $\mu$.
- **Velocity drop detected:** If $\|\dot{z}\|$ decreases while $\eta_{\text{Sch}}$ increases, the correlation confirms capacity-induced slowdown.
- **Persistent high $\eta_{\text{Sch}}$ after fusion:** The interface capacity $C_\partial$ may be the bottleneck. Consider hardware/bandwidth scaling.

**Remediation:**
1. **Ontological Fusion** ({ref}`sec-ontological-fusion-concept-consolidation`): Merge redundant charts to reduce $I_{\text{bulk}}$.
2. **Chart Pruning**: Remove low-utility charts (Definition **Def: Metabolic Pruning Criterion**).
3. **Interface Expansion**: Increase boundary bandwidth (sensor resolution, communication channels).
4. **Depth Reduction**: Decrease TopoEncoder depth to reduce latent dimensionality.

**Computational Proxy:** In practice, compute:

$$
\hat{\eta}_{\text{Sch}} = \frac{|\mathcal{K}| \cdot \bar{H}(z_n | K)}{\log |\mathcal{K}| + d_n \cdot \log(A_{\text{eff}}/\ell_L^2)},

$$
where $|\mathcal{K}|$ is the number of active charts, $\bar{H}(z_n | K)$ is the average conditional entropy of nuisance coordinates, $d_n = \dim(z_n)$, and $A_{\text{eff}}$ is the effective boundary area.



(sec-summary-geometry-bounded-intelligence)=
## Summary: The Geometry of Bounded Intelligence

:::{div} feynman-prose
Let me step back and tell you what we've really learned here.

Intelligence is bounded by geometry. Not by algorithm, not by compute, not by the cleverness of your architecture---by *geometry*. The fundamental limit is the area of your interface, and the fundamental consequence of hitting that limit is paralysis.

This is profound. It says that there's no free lunch in representation. You can't just add more parameters and expect more capability. The parameters have to be grounded through your interface with the world, and that interface has finite bandwidth.

It also tells you what to do when you're stuck. If your agent is slowing down, if learning has plateaued, if the model seems "dead"---check the saturation ratio. You might be approaching the horizon. The solution isn't more training; it's *ontological surgery*---restructuring the representation to reduce bulk information, or expanding the interface to increase capacity.

This is the kind of result that changes how you think about machine learning. It's not just theory---it's engineering guidance. Build agents that respect geometric constraints, and you'll avoid a whole class of failure modes.
:::

**Table 33.6.1 (Causal Information Bound Summary).**

| Concept                      | Definition/Reference                                                                                  | Units         | Diagnostic |
|:-----------------------------|:------------------------------------------------------------------------------------------------------|:--------------|:-----------|
| **Holographic Coefficient**  | $\nu_D = (D-1)\Omega_{D-1}/(8\pi)$ (Def {prf:ref}`def-holographic-coefficient`)                       | dimensionless | —          |
| **Levin Length**             | $\ell_L = \sqrt{\eta_\ell}$ (Def {prf:ref}`def-levin-length`)                                         | $[z]$         | —          |
| **Saturation Limit**         | $I_{\text{bulk}} = C_\partial$ (Def {prf:ref}`def-saturation-limit`)                                  | nat           | Node 40    |
| **Causal Information Bound** | $I_{\max} = \nu_D \cdot \text{Area}(\partial\mathcal{Z})/\ell_L^{D-1}$ (Thm {prf:ref}`thm-causal-information-bound`) | nat           | —          |
| **Saturation Ratio**         | $\eta_{\text{Sch}} = I_{\text{bulk}}/I_{\max}$ (Def {prf:ref}`def-capacity-horizon-diagnostic`)       | dimensionless | Node 56    |
| **Causal Stasis**            | $\|v\|_G \to 0$ as $\eta_{\text{Sch}} \to 1$ (Thm {prf:ref}`thm-causal-stasis`)                       | —             | Node 56    |

**Key Results:**

1. **The Holographic Coefficient** (Definition {prf:ref}`def-holographic-coefficient`) determines how efficiently information can be stored on a boundary of dimension $D$. For $D=2$: $\nu_2 = 1/4$. For $D=3$: $\nu_3 = 1$.

2. **The Levin Length** (Definition {prf:ref}`def-levin-length`) sets the minimal scale of representational distinction. One nat of information occupies $(D-1)$-dimensional volume $\ell_L^{D-1}$.

3. **The Causal Information Bound** (Theorem {prf:ref}`thm-causal-information-bound`) proves that representational capacity is bounded by interface area: $I_{\max} = \nu_D \cdot \text{Area}(\partial\mathcal{Z})/\ell_L^{D-1}$. For the Poincare disk ($D=2$): $I_{\max} = \text{Area}/(4\ell_L)$.

4. **Causal Stasis** (Theorem {prf:ref}`thm-causal-stasis`) shows that approaching this bound causes the agent's internal update rate to vanish. The metric diverges, making belief updates infinitely costly.

5. **Remediation** requires either reducing bulk information (Ontological Fusion) or expanding the interface (hardware scaling). There is no algorithmic workaround.

**Conclusion.** This bound is not a limitation of any particular architecture—it is a fundamental constraint on any agent that must ground its internal representations through a finite-capacity interface. The bound formalizes the intuition that "intelligence" is not a free resource: it must be paid for in interface bandwidth. An agent with finite sensing cannot be infinitely smart—it can only be as smart as its interface is large.



(sec-unified-notation-table-and-cross-section-connectivity)=
## Unified Notation Table and Cross-Section Connectivity

This section provides a consolidated reference for the key symbols introduced across Sections 17-32.

(sec-core-symbols)=
## Core Symbols (Sections 17-32)

| Symbol                         | Name                            | Definition                                                                                                         | Units             | Section        |
|--------------------------------|---------------------------------|--------------------------------------------------------------------------------------------------------------------|-------------------|----------------|
| $G_{ij}(z)$                    | Latent metric tensor            | Capacity-constrained Riemannian metric                                                                             | $[z]^{-2}$        | 2.5, 18.2      |
| $\Gamma^k_{ij}$                | Christoffel symbols             | Levi-Civita connection of $G$                                                                                      | $[z]^{-1}$        | 2.5.1, 22.2.1a |
| $\rho_I(z,t)$                  | Information density             | $-\rho\log\rho + \frac{1}{2}\rho\log\det G$                                                                        | nat$/[z]^n$       | 18.1.2         |
| $C_\partial$                   | Boundary capacity               | Area-law capacity of interface                                                                                     | nat               | 18.1.3         |
| $\nu_{\text{cap}}$             | Capacity saturation             | $I_{\text{bulk}}/C_\partial$                                                                                       | dimensionless     | 18.3.1         |
| $\lambda$                      | WFR length-scale                | Transport-vs-reaction crossover                                                                                    | $[z]$             | 20.2.1, 20.3.1 |
| $\kappa$                       | Screening mass                  | $-\ln\gamma/(c_{\text{info}} \Delta t)$                                                                             | $[z]^{-1}$        | 24.2.4         |
| $\ell_{\text{screen}}$         | Screening length                | $1/\kappa$; reward correlation length                                                                              | $[z]$             | 24.2.4         |
| $U(z)$                         | Hyperbolic potential            | $-2\operatorname{artanh}(\lvert z\rvert)$                                                                          | nat               | 21.1.4         |
| $V(z)$                         | Value/Critic                    | Solution to Helmholtz equation (conservative case)                                                                 | nat               | 2.7, 24.3      |
| $\mathcal{R}$                  | Reward 1-form                   | General reward field; $r_t = \langle\mathcal{R}, v\rangle$                                                         | nat$/[z]$         | 24.1           |
| $\mathcal{F}$                  | Value Curl                      | $d\mathcal{R}$; measures non-conservative structure                                                                | nat$/[z]^2$       | 24.2           |
| $\Phi$                         | Scalar Potential                | Hodge gradient component of $\mathcal{R}$                                                                          | nat               | 24.2           |
| $\Psi$                         | Vector Potential                | Hodge solenoidal component of $\mathcal{R}$                                                                        | nat$\cdot[z]^2$   | 24.2           |
| $\eta$                         | Harmonic Flux                   | Hodge harmonic component of $\mathcal{R}$                                                                          | nat$/[z]$         | 24.2           |
| $\mathbf{A}$                   | Vector Potential (WFR)          | $d\mathbf{A} = \mathcal{F}$; appears in generalized WFR action                                                     | nat$/[z]$         | 20.2           |
| $\beta_{\text{curl}}$          | Curl Coupling                   | Lorentz force strength                                                                                             | dimensionless     | 22.2           |
| $J$                            | Probability Current             | $\rho v - D\nabla\rho$; non-zero in NESS                                                                           | $1/\text{step}$   | 24.4           |
| $\Phi_{\text{eff}}$            | Effective potential             | $\alpha U + (1-\alpha)\Phi + \gamma_{\text{risk}}\Psi_{\text{risk}}$                                               | nat               | 22.3.1         |
| $u_\pi$                        | Control field                   | Policy-induced tangent vector                                                                                      | $[z]/\text{step}$ | 21.2.2         |
| $T_c$                          | Cognitive temperature           | Exploration parameter                                                                                              | nat               | 22.4           |
| $\Omega(z)$                    | Conformal factor                | $1 + \alpha_{\text{conf}}\lVert\nabla^2 V\rVert$                                                                   | dimensionless     | 24.4.1         |
| $\omega$                       | Symplectic form                 | $\sum_i dq^i \wedge dp_i$                                                                                          | nat               | 23.1.1         |
| $\mathcal{L}$                  | Legendre transform              | $T\mathcal{Q} \to T^*\mathcal{Q}$; $p = G\dot{q}$                                                                  | —                 | 23.2.3         |
| $\mathcal{M}_\Theta$           | Parameter manifold              | Space of agent parameters                                                                                          | —                 | 26.2           |
| $\Psi$                         | Constraint evaluation map       | $\theta \mapsto [C_1(\theta), \ldots, C_K(\theta)]$                                                                | —                 | 26.3           |
| $\pi_{\mathfrak{G}}$           | Governor policy                 | $s_{t:t-H} \mapsto \Lambda_t$                                                                                      | —                 | 26.3           |
| $V_{\mathfrak{L}}$             | Training Lyapunov               | $\mathcal{L} + \sum_k \frac{\mu_k}{2}\max(0,C_k)^2$                                                                | nat               | 26.5           |
| $\gamma_{\text{viol}}$         | Violation penalty               | Constraint violation weight                                                                                        | dimensionless     | 26.4           |
| $\Lambda_t$                    | Control vector                  | $(\eta_t, \vec{\lambda}_t, T_{c,t})$                                                                               | mixed             | 26.3           |
| $\Xi_T$                        | Memory screen                   | $\int_0^T \alpha(t') \delta_{\gamma(t')} dt'$                                                                      | nat               | 27.1.2         |
| $H_\tau(z, z')$                | Heat kernel                     | Memory kernel (fundamental soln to heat eqn)                                                                       | $[z]^{-d}$        | 27.2.1         |
| $\tau$                         | Diffusion time                  | Memory smoothing scale                                                                                             | $[z]^2$           | 27.2.1         |
| $\Psi_{\text{mem}}$            | Memory potential                | $-\int H_\tau(z, z') d\Xi_T(z')$                                                                                   | nat               | 27.2.2         |
| $\Omega_{\text{mem}}$          | Non-locality ratio              | $\lVert\nabla_G \Psi_{\text{mem}}\rVert_G / \lVert\nabla_G \Phi_{\text{eff}}\rVert_G$                              | dimensionless     | 27.5.1         |
| $\mathcal{Z}^{(N)}$            | N-agent product manifold        | $\prod_{i=1}^N \mathcal{Z}^{(i)}$                                                                                  | $[z]$             | 29.1           |
| $\mathcal{B}_{ij}$             | Bridge manifold                 | Interaction submanifold between agents $i,j$                                                                       | $[z]$             | 29.2           |
| $\Phi_{ij}$                    | Strategic potential             | Interaction kernel from agent $j$                                                                                  | nat               | 29.3           |
| $\mathcal{G}_{ij}^{kl}$        | Game Tensor                     | $\partial^2 V^{(i)} / \partial z^{(j)}_k \partial z^{(j)}_l$                                                       | nat$/[z]^2$       | 29.4           |
| $\tilde{G}^{(i)}$              | Game-augmented metric           | $G^{(i)} + \alpha_{\text{adv}} \mathcal{G}_{ij}$                                                                   | $[z]^{-2}$        | 29.4           |
| $\epsilon_{\text{Nash}}$       | Nash residual                   | Max gradient deviation from equilibrium                                                                            | nat$/[z]$         | 29.6           |
| $\emptyset$                    | Semantic Vacuum                 | Fiber over origin $z=0$; maximal $SO(D)$ symmetry                                                                  | —                 | 30.1           |
| $\Xi$                          | Ontological Stress              | $I(z_{\text{tex},t}; z_{\text{tex},t+1} \mid K_t, z_{n,t}, K^{\text{act}}_t)$                                                   | nat               | 30.2           |
| $\Xi_{\text{crit}}$            | Fission threshold               | Critical stress for chart bifurcation                                                                              | nat               | 30.3           |
| $\mathcal{L}_{\text{center}}$  | Centering loss                  | $\lVert\sum q_i\rVert^2 + \sum\lVert\sum e_{i,c}\rVert^2$                                                          | $[z]^2$           | 30.1           |
| $\mathcal{L}_{\text{Ricci}}$   | Ricci flow loss                 | $\lVert R_{ij} - \frac{1}{2}RG_{ij} + \Lambda G_{ij} - \kappa T_{ij}\rVert_F^2 + \nu^2\lVert\nabla^2\Xi\rVert_F^2$ | $[z]^{-4}$        | 30.5           |
| $\Upsilon_{ij}$                | Ontological Redundancy          | $\exp(-[d_{\text{WFR}} + D_{\mathrm{KL}} + \lVert V_i - V_j\rVert^2])$                                             | dimensionless     | 30.8           |
| $G_\Delta$                     | Discrimination Gain             | $I(X; \{K_i, K_j\}) - I(X; K_{i \cup j})$                                                                          | nat               | 30.8           |
| $\Upsilon_{\text{crit}}$       | Fusion threshold                | Critical redundancy for chart merger                                                                               | dimensionless     | 30.9           |
| $\epsilon_{\text{hysteresis}}$ | Hysteresis constant             | Fission/Fusion asymmetry term                                                                                      | nat               | 30.9           |
| $\sigma_k^2$                   | Intra-Symbol Variance           | $\mathbb{E}[\lVert z_e - e_k\rVert^2 \mid K=k]$                                                                    | $[z]^2$           | 30.12          |
| $\mathcal{D}_f$                | Functional Indistinguishability | $D_{\mathrm{KL}}(\pi_1 \lVert \pi_2) + \lVert V_1 - V_2\rVert$                                                     | nat               | 30.12          |
| $\mathcal{V}_k$                | Voronoi cell                    | $\{z : d_G(z, e_k) \le d_G(z, e_j)\}$                                                                              | —                 | 30.12          |
| $\mathcal{D}_k$                | Local Distortion                | $\int_{\mathcal{V}_k} d_G(z, e_k)^2 p(z) d\mu_G$                                                                   | $[z]^2$           | 30.12          |
| $U_k$                          | Symbol Utility                  | $P(k) \cdot I(K=k; A) + P(k) \cdot I(K=k; K_{t+1})$                                                                | nat               | 30.12          |
| $\dot{\mathcal{M}}(s)$         | Metabolic flux                  | WFR action rate (transport + reaction cost)                                                                        | nat/step          | 31.1           |
| $\Psi_{\text{met}}(s)$         | Metabolic potential             | Cumulative dissipation $\int_0^s \dot{\mathcal{M}} \, du$                                                          | nat               | 31.2           |
| $\mathcal{S}_{\text{delib}}$   | Deliberation action             | $-\langle V \rangle_{\rho_S} + \Psi_{\text{met}}(S)$                                                               | nat               | 31.2           |
| $S^*$                          | Optimal computation budget      | Deliberation stopping time                                                                                         | step              | 31.3           |
| $\Gamma(s)$                    | Value-Improvement Rate          | $\lVert d\langle V \rangle/ds\rVert$                                                                               | nat/step          | 31.3           |
| $\sigma_{\text{tot}}$          | Total entropy production        | $\dot{H} + \dot{\mathcal{M}}/T_c \ge 0$                                                                            | nat/step          | 31.4           |
| $\eta_{\text{thought}}$        | Efficiency of thought           | $-T_c \dot{H}/\dot{\mathcal{M}} \le 1$                                                                             | dimensionless     | 31.4           |
| $\mathfrak{I}$                 | Interventional operator         | Pearl's $do(\cdot)$ surgery                                                                                        | —                 | 32.1           |
| $\Psi_{\text{causal}}$         | Causal information potential    | EIG for transition parameters                                                                                      | nat               | 32.2           |
| $\Delta_{\text{causal}}$       | Causal deficit                  | $D_{\text{KL}}(P_{\text{int}} \lVert P_{\text{obs}})$                                                              | nat               | 32.2           |
| $\mathbf{f}_{\text{exp}}$      | Curiosity force                 | $G^{-1}\nabla\Psi_{\text{causal}}$                                                                                 | $[z]$/step        | 32.3           |
| $\beta_{\text{exp}}$           | Exploration coefficient         | Curiosity vs. exploitation balance                                                                                 | dimensionless     | 32.3           |
| $\nu_D$                        | Holographic Coefficient         | $(D-1)\Omega_{D-1}/(8\pi)$; dim-dependent capacity factor                                                          | dimensionless     | 33.0           |
| $\ell_L$                       | Levin Length                    | $\sqrt{\eta_\ell}$; minimal distinction scale                                                                      | $[z]$             | 33.1           |
| $I_{\max}$                     | Causal Information Bound        | $\nu_D \cdot \text{Area}(\partial\mathcal{Z})/\ell_L^{D-1}$                                                        | nat               | 33.3           |
| $\eta_{\text{Sch}}$            | Saturation Ratio                | $I_{\text{bulk}}/I_{\max}$                                                                                         | dimensionless     | 33.5           |
| $r_h$                          | Horizon radius                  | Critical radius where $G_{rr} \to \infty$                                                                          | $[z]$             | 33.2           |

(sec-boundary-conditions)=
## Boundary Conditions ({ref}`sec-the-boundary-interface-symplectic-structure`)

| Type      | Symbol                                               | Interpretation              | Physics             |
|-----------|------------------------------------------------------|-----------------------------|---------------------|
| Dirichlet | $\rho\lvert_{\partial} = \delta(q - q_{\text{obs}})$ | Position clamped by sensors | Environment → Agent |
| Neumann   | $\nabla_n\rho = j_{\text{motor}}$                    | Flux clamped by motors      | Agent → Environment |
| Source    | $J_r$                                                | Reward flux on boundary     | Reward signal       |

(sec-cross-section-connectivity-map)=
## Cross-Section Connectivity Map (Sections 17-32)

```
{ref}`sec-summary-unified-information-theoretic-control-view` (Summary)
     |
     v
{ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits` (Capacity Law) ─────────────────────────────────────────┐
     |                                                              |
     | $\rho_I$, $C_\partial$, $T_{ij}$                            |
     v                                                              |
Section 19 (Conclusion) ←───────────────────────────────────────────────────────┐  |
     |                                                           |  |
     v                                                           |  |
{ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces` (WFR Geometry) ──────────────────────────────────┐   |  |
     |                                                       |   |  |
     | $\lambda$, $(v, r)$, WFR metric                       |   |  |
     v                                                       |   |  |
{ref}`sec-radial-generation-entropic-drift-and-policy-control` (Holographic Generation {cite}`thooft1993holographic,susskind1995world`) ────────────────┐       |   |  |
     |                                               |       |   |  |
     | $U(z)$, $u_\pi$, SO(D) breaking              |       |   |  |
     v                                               v       v   |  |
{ref}`sec-the-equations-of-motion-geodesic-jump-diffusion` (Equations of Motion) ←──────────────────┴───────┴───┘  |
     |                                                              |
     | $\Phi_{\text{eff}}$, geodesic SDE, BAOAB                    |
     v                                                              |
{ref}`sec-the-boundary-interface-symplectic-structure` (Holographic Interface) ←────────────────────────────────┤
     |                                                              |
     | Symplectic structure, Legendre transform, $(q, p)$          |
     v                                                              |
{ref}`sec-the-reward-field-value-forms-and-hodge-geometry` (Scalar Field) ←─────────────────────────────────────────┘
     |
     | $V$ as Helmholtz solution, conformal coupling $\Omega$
     v
{ref}`sec-supervised-topology-semantic-potentials-and-metric-segmentation` (Supervised Topology)
     |
     | Classification as geodesic relaxation
     v
{ref}`sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller` (Meta-Stability) ←─────────────────────── {ref}`sec-adaptive-multipliers-learned-penalties-setpoints-and-calibration`
     |                                               (Adaptive Multipliers)
     | $\pi_{\mathfrak{G}}$, $V_{\mathfrak{L}}$, bilevel optimization
     v
{ref}`sec-section-non-local-memory-as-self-interaction-functional` (Non-Local Memory) ←──────────────────── {ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`, 22, 24
     |                                               (WFR, EoM, Scalar Field)
     | $\Xi_T$, $H_\tau$, $\Psi_{\text{mem}}$, $\Omega_{\text{mem}}$
     v
{ref}`sec-section-hyperbolic-active-retrieval-geodesic-search-and-semantic-pull-back` (Hyperbolic Retrieval) ←────────────────── {ref}`sec-radial-generation-entropic-drift-and-policy-control` (Poincare metric)
     |                                               {ref}`sec-section-non-local-memory-as-self-interaction-functional` (Memory potential)
     | $\Phi_{\text{ret}}$, Geodesic search, WFR sources
     v
{ref}`sec-symplectic-multi-agent-field-theory` (Multi-Agent SMFT) ←──────────────────── {ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits`, 21, 23
     |                                               (Metric, Symplectic, Capacity)
     | $\mathcal{G}_{ij}$, Strategic potential, Nash equilibrium
     v
{ref}`sec-ontological-expansion-topological-fission-and-the-semantic-vacuum` (Ontological Expansion) ←───────────────── {ref}`sec-radial-generation-entropic-drift-and-policy-control` (Pitchfork bifurcation)
     |                                               {ref}`sec-tier-the-attentive-atlas` (Attentive Atlas)
     | $\Xi$, $\emptyset$, Query Fission, Ricci flow   {ref}`sec-main-result` (Metric law)
     v
Appendices (Derivations, Units, WFR Tensor)
```

(sec-diagnostic-node-registry)=
## Diagnostic Node Registry (Complete)

| #  | Name                                              | Section | Key Formula                                                                                                      |
|----|---------------------------------------------------|---------|------------------------------------------------------------------------------------------------------------------|
| 1  | [CostBoundCheck](#sec-the-stability-checks)       | 3.5     | $\max(0, V(z) - V_{\text{max}})^2$                                                                               |
| 2  | [ZenoCheck](#sec-the-stability-checks)            | 3.5     | $D_{\mathrm{KL}}(\pi_t \Vert \pi_{t-1})$                                                                         |
| 3  | [CompactCheck](#sec-the-stability-checks)         | 3.5     | $H(q(K \mid x))$                                                                                                 |
| 4  | [ScaleCheck](#sec-the-stability-checks)           | 3.5     | $\lVert \nabla \theta \rVert / \lVert \Delta S \rVert$                                                           |
| 5  | [ParamCheck](#sec-the-stability-checks)           | 3.5     | $\lVert \nabla_t S_t \rVert^2$                                                                                   |
| 6  | [GeomCheck](#sec-the-stability-checks)            | 3.5     | $\mathcal{L}_{\text{contrastive}}$ (InfoNCE)                                                                     |
| 7  | [StiffnessCheck](#sec-the-stability-checks)       | 3.5     | $\max(0, \epsilon - \lVert \nabla_A V \rVert)$                                                                     |
| 8  | [TopoCheck](#sec-the-stability-checks)            | 3.5     | $T_{\text{reach}}(z_{\text{goal}})$                                                                              |
| 9  | [TameCheck](#sec-the-stability-checks)            | 3.5     | $\lVert \nabla^2 S_t \rVert$                                                                                     |
| 10 | [ErgoCheck](#sec-the-stability-checks)            | 3.5     | $-H(\pi)$                                                                                                        |
| 11 | [ComplexCheck](#sec-the-stability-checks)         | 3.5     | $H(K)/\log\lvert\mathcal{K}\rvert$                                                                               |
| 12 | [OscillateCheck](#sec-the-stability-checks)       | 3.5     | $\lVert z_t - z_{t-2} \rVert$                                                                                    |
| 13 | [BoundaryCheck](#sec-the-stability-checks)        | 3.5     | $I(X;K)$                                                                                                         |
| 14 | [InputSaturationCheck](#sec-the-stability-checks) | 3.5     | $\mathbb{I}(\lvert x \rvert > x_{\text{max}})$                                                                   |
| 15 | [SNRCheck](#sec-the-stability-checks)             | 3.5     | $\text{SNR} < \epsilon$                                                                                          |
| 16 | [AlignCheck](#sec-the-stability-checks)           | 3.5     | $\lvert V_{\text{proxy}} - V_{\text{true}} \rvert$                                                               |
| 17 | [Lock](#sec-the-stability-checks)                 | 3.5     | $\mathbb{I}(\text{Unsafe}) \cdot \infty$                                                                         |
| 18 | [SymmetryCheck](#sec-the-stability-checks)        | 3.5     | $\mathbb{E}_{g\sim G}[D_{\mathrm{KL}}(q(K\mid x)\Vert q(K\mid g\cdot x))]$                                       |
| 19 | [DisentanglementCheck](#sec-the-stability-checks) | 3.5     | $\lVert\mathrm{Cov}(z_{\text{macro}},z_n)\rVert_F^2$                                                             |
| 20 | [LipschitzCheck](#sec-the-stability-checks)       | 3.5     | $\max_\ell \sigma(W_\ell)$                                                                                       |
| 21 | [SymplecticCheck](#sec-the-stability-checks)      | 3.5     | $\lVert J_S^\top J J_S - J\rVert_F^2$                                                                            |
| 22 | [MECCheck](#sec-the-stability-checks)             | 3.5     | $\lVert(\varrho_{t+1}-\varrho_t)/\Delta t - \mathcal{L}_{\text{GKSL}}(\varrho_t)\rVert_F^2$                      |
| 23 | [NEPCheck](#sec-the-stability-checks)             | 3.5     | $\mathrm{ReLU}(D_{\mathrm{KL}}(p_{t+1}\Vert p_t)-I(X_t;K_t))^2$                                                  |
| 24 | [QSLCheck](#sec-the-stability-checks)             | 3.5     | $\mathrm{ReLU}(d_G(z_{t+1},z_t)-v_{\max})^2$                                                                     |
| 25 | [HoloGenCheck](#node-25)                          | 21.4    | $\mathbf{1}(\lVert z\rVert \geq R_{\text{cutoff}})$                                                              |
| 26 | [GeodesicCheck](#node-26)                         | 22.6    | $\lVert\ddot{z} + \Gamma(\dot{z},\dot{z}) + G^{-1}\nabla\Phi_{\text{eff}} - u_\pi - \beta_{\text{curl}} G^{-1}\mathcal{F}\dot{z}\rVert_G$ |
| 27 | [OverdampedCheck](#node-27)                       | 22.6    | $\gamma / \lVert \mathcal{M}_\gamma^{-1}\,v\rVert$             |
| 28 | [JumpConsistencyCheck](#node-28)                  | 22.6    | $\lVert m_{\text{pre}} - m_{\text{post}}\eta\rVert$                                                              |
| 29 | [TextureFirewallCheck](#node-29)                  | 22.6    | $\lVert\partial_{z_{\text{tex}}} \dot{z}\rVert$                                                                  |
| 30 | [SymplecticBoundaryCheck](#node-30)               | 23.8    | $\lVert E_\phi(x) - q_{\text{clamp}}\rVert_G$                                                                    |
| 31 | [DualAtlasConsistencyCheck](#node-31)             | 23.8    | $\lVert D_A(E_A(a)) - a\rVert$                                                                                   |
| 32 | [MotorTextureCheck](#node-32)                     | 23.8    | $H(z_{\text{tex,motor}} \mid A, z_{n,\text{motor}})$                                                             |
| 33 | [ThermoCycleCheck](#node-33)                      | 23.8    | $\lVert\Delta S_{\text{cycle}}\rVert$                                                                            |
| 34 | [ContextGroundingCheck](#node-34)                 | 23.8    | $I(c; z)$                                                                                                        |
| 35 | [HelmholtzResidualCheck](#node-35)                | 24.7    | $\lVert-\Delta_G V + \kappa^2 V - \rho_r\rVert$                                                                  |
| 36 | [GreensFunctionDecayCheck](#node-36)              | 24.7    | $\lVert V(z) - V(z')\rVert \cdot e^{\kappa d_G(z,z')}$                                                           |
| 37 | [BoltzmannConsistencyCheck](#node-37)             | 24.7    | $D_{\mathrm{KL}}(P_{\text{empirical}} \lVert P_{\text{Boltzmann}})$                                              |
| 38 | [ConformalBackReactionCheck](#node-38)            | 24.7    | $\text{Var}(\Omega)$                                                                                             |
| 39 | [ValueMassCorrelationCheck](#node-39)             | 24.7    | $\text{corr}(m_t, V(z_t))$                                                                                       |

Here $v := \dot{z}$ and $\mathcal{M}_\gamma^{-1} = \gamma I - \beta_{\text{curl}} G^{-1}\mathcal{F}$.
| 40 | [CapacitySaturationCheck](#node-40)               | 18.3    | $I_{\text{bulk}}/C_\partial$                                                                                     |
| 41 | [SupervisedTopologyChecks](#node-41)              | 25.4    | (See {ref}`sec-the-supervised-topology-loss`)                                                                                               |
| 42 | [GovernorStabilityCheck](#node-42)                | 26.9    | $\Delta V_{\mathfrak{L}} = V_{\mathfrak{L}}(\theta_{t+1}) - V_{\mathfrak{L}}(\theta_t)$                          |
| 43 | [MemoryBalanceCheck](#node-43)                    | 27.5    | $\Omega_{\text{mem}} = \lVert\nabla_G\Psi_{\text{mem}}\rVert_G / \lVert\nabla_G\Phi_{\text{eff}}\rVert_G$        |
| 44 | [HyperbolicAlignmentCheck](#node-44)              | 28.6    | $\Delta_{\text{align}} := \mathbb{E}[\lVert d_{\mathbb{D}}^{\text{int}} - d_{\mathbb{D}}^{\text{ext}}\rVert]$    |
| 45 | [RetrievalFirewallCheck](#node-45)                | 28.6    | $\Gamma_{\text{leak}} := \lVert\nabla_{z_{\text{int}}} (\partial \pi / \partial z_{\text{tex,ext}})\rVert$       |
| 46 | [GameTensorCheck](#node-46)                       | 29.6    | $\lVert\mathcal{G}_{ij}\rVert_F$                                                                                 |
| 47 | [NashResidualCheck](#node-47)                     | 29.6    | $\epsilon_{\text{Nash}} := \max_i \lVert(G^{(i)})^{-1}\nabla \Phi_{\text{eff}}^{(i)}\rVert_{G^{(i)}}$            |
| 48 | [SymplecticBridgeCheck](#node-48)                 | 29.6    | $\Delta_\omega := \lVert\int_{\mathcal{B}_{ij}} \omega_{ij}(t) - \omega_{ij}(0)\rVert$                           |
| 49 | [OntologicalStressCheck](#node-49)                | 30.6    | $\Xi := I(z_{\text{tex},t}; z_{\text{tex},t+1} \mid K_t, z_{n,t}, K^{\text{act}}_t)$                                          |
| 50 | [FissionReadinessCheck](#node-50)                 | 30.6    | $\mathbb{I}(\Xi > \Xi_{\text{crit}}) \cdot \mathbb{I}(\Delta V_{\text{proj}} > \mathcal{C}_{\text{complexity}})$ |
| 51 | [MetabolicEfficiencyCheck](#node-51)              | 31.5    | $\eta_{\text{ROI}} := \lvert\Delta\langle V\rangle\rvert / \Psi_{\text{met}}(S)$                                 |
| 52 | [EntropyProductionCheck](#node-52)                | 31.5    | $\sigma_{\text{tot}} := \dot{H} + \dot{\mathcal{M}}/T_c \ge 0$                                                   |
| 53 | [CausalEnclosureCheck](#node-53)                  | 32.6    | $\Delta_{\text{causal}} < \delta_{\text{causal}}$                                                                |
| 54 | {prf:ref}`node-fusion-readiness-check`                  | 30.11   | $\max_{i \neq j} \Upsilon_{ij} > \Upsilon_{\text{crit}}$                                                         |
| 55 | {prf:ref}`node-codebook-liveness-check`                 | 30.11   | $\min_k P(K=k) < \epsilon_{\text{dead}}$                                                                         |
| 56 | [CapacityHorizonCheck](#node-56)                  | 33.5    | $\eta_{\text{Sch}} := I_{\text{bulk}} / I_{\max}$                                                                |
| 61 | [ValueCurlCheck](#node-61)                        | 24.8    | $\oint_\gamma \delta_{\text{TD}} \approx \int\lVert\nabla\times\mathcal{R}\rVert$                                |
