(sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces)=
# Wasserstein-Fisher-Rao Geometry: Unified Transport on Hybrid State Spaces

## TLDR

- When state is hybrid (discrete $K$ + continuous $z_n$), belief evolution requires a geometry that handles both **flow**
  (transport within charts) and **jumps** (reaction between charts).
- The Wasserstein–Fisher–Rao (WFR / Hellinger–Kantorovich) metric gives a **single variational principle** for this
  hybrid belief dynamics.
- WFR resolves “duct-tape” product-metric heuristics by pricing transport and reaction consistently, and it yields
  implementable consistency diagnostics (WFRCheck).
- Think operationally: belief is a **fluid** that can move and transform; WFR measures the cheapest way to do both.
- This chapter connects the metric law (capacity) to dynamics (geodesic jump-diffusion) and to the information bound
  (area-law-like limits).

## Roadmap

1. Why product metrics fail for hybrid state spaces.
2. Define WFR and interpret transport vs. reaction components.
3. Connect WFR to filtering/control objectives and to implementation diagnostics.

:::{div} feynman-prose
Let me tell you about one of the most elegant solutions I've ever seen to a problem that seems hopelessly messy at first.

Here's the situation. Our agent has an internal representation that mixes discrete and continuous parts. The discrete part says "what kind of situation is this?"---are we in the kitchen or the living room, is the object a cup or a bottle? The continuous part says "where exactly within that situation?"---the precise position, orientation, all the fine details.

Now, the standard approach is to handle these separately. You have some kind of graph or finite-state machine for the discrete part, and a Riemannian manifold for the continuous part, and you glue them together with duct tape and hope for the best. When does the agent "jump" from one discrete state to another? How do you compare paths that involve different combinations of jumping and moving? The whole thing becomes a computational and conceptual nightmare.

But there's a beautiful way out of this mess, and it comes from a surprising place: optimal transport theory. The key insight is deceptively simple: stop thinking about the agent's state as a *point* that moves around, and start thinking about it as a *distribution* that flows and transforms.
:::

The latent bundle $\mathcal{Z} = \mathcal{K} \times \mathcal{Z}_n \times \mathcal{Z}_{\mathrm{tex}}$ ({ref}`Section 2.2a <sec-the-trinity-of-manifolds>`) combines a discrete macro-state $K$ with continuous nuisance coordinates $z_n$. The product metric $d_{\mathcal{K}} \oplus G_n$ (Definition 2.2.1) and the Sasaki-like warped metric ({ref}`Section 7.11.3 <sec-the-induced-riemannian-geometry>`) were heuristic constructions that treat the discrete and continuous components separately. These constructions are constrained by the agent's {prf:ref}`def-boundary-markov-blanket`.

(rb-distribution-shift)=
:::{admonition} Researcher Bridge: Handling Distribution Shift
:class: info
Standard Bayesian filters fail during "surprises" because they can't handle mass appearing or disappearing (Unbalanced Transport). The **Wasserstein-Fisher-Rao (WFR)** metric allows the agent's belief to both **flow** (smooth tracking) and **jump** (teleporting probability mass). This provides a unified variational principle for both continuous state-tracking and discrete hypothesis-switching.
:::

This section introduces the **Wasserstein-Fisher-Rao (WFR)** metric---also known as **Hellinger-Kantorovich** {cite}`chizat2018unbalanced,liero2018optimal`---which provides a rigorous, unified variational principle. The key insight is to treat the agent's internal state not as a *point* in $\mathcal{Z}$, but as a *measure* (belief state) $\rho_s \in \mathcal{M}^+(\mathcal{Z})$ evolving on the bundle.

(sec-motivation-the-failure-of-product-metrics)=
## Motivation: The Failure of Product Metrics

:::{div} feynman-prose
Before we dive into the solution, let's make sure we really understand the problem. Why doesn't the obvious approach work?

The obvious approach is what I call the "Sasaki-like construction"---you take the metric on the discrete part, you take the metric on the continuous part, and you combine them. It's like saying: the distance between two states is the discrete hop distance plus the continuous Riemannian distance.

This seems reasonable, but watch what happens when things get interesting.
:::

**The Problem with Sasaki-like Constructions.**

The metric tensor from {ref}`Section 7.11.3 <sec-the-induced-riemannian-geometry>` (where $\rho_{\text{depth}}$ denotes resolution depth, not density):

$$
ds^2 = d\rho_{\text{depth}}^2 + d\sigma_{\mathcal{K}}^2 + e^{-2\rho_{\text{depth}}}\|dz_n\|^2

$$
assumes a fixed point moving through the bundle. This creates two problems:

1. **Discontinuous Jumps:** When the agent transitions from chart $K_i$ to chart $K_j$, the metric provides no principled way to measure the "cost" of the jump versus continuous motion along an overlap.

2. **No Mass Conservation:** A point either is or isn't at a location. But the agent's *belief* can be partially in multiple charts simultaneously (soft routing, {ref}`Section 7.8 <sec-tier-the-attentive-atlas>`).

:::{div} feynman-prose
Let me make this very concrete. Suppose the agent is tracking an object that suddenly moves behind an occluder. The belief distribution should smoothly transition from "I'm pretty sure where it is" to "it could be in several places." But if we're tracking a *point*, we have to decide: does the point stay where it was, or does it jump? Neither option is right---the situation calls for a distribution that spreads out.

Or consider this: the agent is 90% confident it's in scenario A and 10% confident it's in scenario B. What's the "position" of that belief? There isn't one! You need a distribution.
:::

:::{admonition} The Core Problem
:class: warning feynman-added
Think of a particle versus a probability cloud. A particle has to be *somewhere*---it can move, but it can't be in two places at once. A probability cloud can spread, concentrate, flow, and even split. The agent's belief is fundamentally a cloud, not a particle. Treating it as a particle forces artificial discretization: when do you "switch" hypotheses? The WFR framework says: you don't have to choose. Mass can continuously redistribute.
:::

**The WFR Solution.**

The Wasserstein-Fisher-Rao metric resolves both issues by lifting dynamics to the space of measures $\mathcal{M}^+(\mathcal{Z})$. In this space:
- **Transport (Wasserstein):** Probability mass moves along continuous coordinates via the continuity equation.
- **Reaction (Fisher-Rao):** Probability mass is created/annihilated locally, enabling discrete chart transitions.

The metric determines the optimal path by minimizing the total cost: transport cost $\int\|v\|_G^2\,d\rho$ plus reaction cost $\int\lambda^2|r|^2\,d\rho$.

:::{div} feynman-prose
Here's the beautiful idea. Instead of asking "where is the agent's belief *point*?", we ask "what is the agent's belief *distribution*?" And instead of asking "how does the point move?", we ask "how does the distribution evolve?"

This distribution can do two things: it can *flow* (mass moves from here to there while conserving total probability) or it can *react* (mass appears or disappears locally). The first is what happens when you track a moving object. The second is what happens when you suddenly realize "wait, I was wrong about which scenario I'm in."

The WFR metric gives us a principled way to measure the "cost" of any combination of flowing and reacting. And here's the punchline: finding the optimal path in this space of distributions turns out to be a convex optimization problem. No more combinatorial explosion. No more arbitrary choices about when to "jump."
:::

(sec-the-wfr-metric)=
## The WFR Metric (Benamou-Brenier Formulation)

:::{div} feynman-prose
Now let's get precise. The Benamou-Brenier formulation is a beautiful way to think about optimal transport: instead of asking "what's the cheapest way to rearrange mass from configuration A to configuration B?", you ask "what's the most efficient *process* that transforms A into B over time?"

It's like the difference between asking "what's the shortest path between two cities?" and asking "what's the most fuel-efficient way to drive between them, considering traffic and terrain?" The second question embeds the problem in time and lets you think about dynamics.
:::

Let $\rho(s, z)$ be a time-varying density on the latent bundle $\mathcal{Z}$, evolving in computation time $s$. The WFR distance is defined by the minimal action of a generalized continuity equation.

:::{prf:definition} The Generalized WFR Action
:label: def-the-wfr-action

The squared WFR distance $d^2_{\mathrm{WFR}}(\rho_0, \rho_1)$ is the infimum of the generalized energy functional:

$$
\mathcal{E}[\rho, v, r] = \int_0^1 \int_{\mathcal{Z}} \left( \underbrace{\|v_s(z)\|_G^2}_{\text{Transport Cost}} + \underbrace{\lambda^2 |r_s(z)|^2}_{\text{Reaction Cost}} - \underbrace{2\langle \mathbf{A}(z), v_s(z) \rangle}_{\text{Vector Potential}} \right) d\rho_s(z) \, ds

$$
subject to the **Unbalanced Continuity Equation**:

$$
\partial_s \rho + \nabla \cdot (\rho v) = \rho r

$$
where:
- $v_s(z) \in T_z\mathcal{Z}$ is the **velocity field** (transport/flow)
- $r_s(z) \in \mathbb{R}$ is the **reaction rate** (growth/decay of mass)
- $\lambda > 0$ is the **length-scale parameter** balancing transport and reaction
- $G$ is the Riemannian metric on the continuous fibres ({ref}`Section 2.5 <sec-second-order-sensitivity-value-defines-a-local-metric>`)
- $\mathbf{A}(z)$ is the **vector potential** satisfying $d\mathbf{A} = \mathcal{F}$ (the {prf:ref}`def-value-curl`)

*Units:* $[\mathbf{A}] = \mathrm{nat}/[\text{length}]$.

**Conservative Limit:** When $\mathcal{F} = 0$ (Definition {prf:ref}`def-conservative-reward-field`), we can choose the gauge $\mathbf{A} = 0$ and recover the standard WFR action without the vector potential term.

**Non-Conservative Case:** When $\mathcal{F} \neq 0$, the vector potential term couples the transport velocity to the solenoidal component of the reward field. The Euler-Lagrange equations of this action yield the Lorentz-Langevin equation (Definition {prf:ref}`def-bulk-drift-continuous-flow`).

*Remark (Gauge Invariance).* The action is invariant under gauge transformations $\mathbf{A} \to \mathbf{A} + d\chi$ for any scalar $\chi$, since $d(d\chi) = 0$. We fix the gauge via the Coulomb condition $\delta\mathbf{A} = 0$ (divergence-free).

*Forward reference (Boundary Conditions).* {ref}`Section 23.5 <sec-wfr-boundary-conditions-waking-vs-dreaming>` specifies how boundary conditions on $\partial\mathcal{Z}$ (sensory and motor boundaries) constrain the WFR dynamics: **Waking** imposes Dirichlet (sensors) + Neumann (motors) BCs; **Dreaming** imposes reflective BCs on both, enabling recirculating flow without external input.

:::

:::{div} feynman-prose
Let me unpack this piece by piece, because there's a lot going on.

The **unbalanced continuity equation** is the heart of the matter: $\partial_s \rho + \nabla \cdot (\rho v) = \rho r$. On the left side, we have how the density changes with time ($\partial_s \rho$) plus how it flows due to velocity ($\nabla \cdot (\rho v)$). On the right side, we have the reaction term ($\rho r$)---if $r > 0$, mass is being created; if $r < 0$, mass is being destroyed.

In ordinary optimal transport, the right side is zero: mass is conserved, it just moves around. That's the "balanced" case. But we need the "unbalanced" case because when an agent switches hypotheses---goes from "I think it's scenario A" to "I think it's scenario B"---mass has to disappear from A and appear in B. That's not transport; that's reaction.

The **action functional** measures the total cost of a path. You pay for velocity (moving mass around) and you pay for reaction (creating or destroying mass). The parameter $\lambda$ sets the exchange rate: how much is one unit of transport worth compared to one unit of reaction?

And that **vector potential** term? That's for situations where the reward landscape has "curl"---where going around in a circle doesn't bring you back to the same value. In that case, the optimal path isn't just about minimizing distance; it's about exploiting the curl, like a sailor tacking against the wind.
:::

:::{admonition} Example: Belief Update as WFR Flow
:class: feynman-added example

Imagine a robot tracking a ball. Initially, the belief is concentrated near position $x_0$. Then the ball moves quickly to $x_1$. What happens to the belief?

**Pure transport ($r = 0$):** The belief distribution flows smoothly from $x_0$ to $x_1$. This is what happens during normal tracking when the ball moves predictably.

**Pure reaction ($v = 0$):** The belief at $x_0$ shrinks while belief at $x_1$ grows. This is what happens during a "surprise"---the ball teleports (occlusion, fast motion), and rather than flowing smoothly, the belief essentially jumps.

**Mixed:** Usually both happen. The belief flows toward where you expect the ball to go, but also mass is transferred to alternative hypotheses ("maybe it bounced off something I didn't see").

The WFR metric finds the optimal mix. If $x_1$ is close to $x_0$, transport dominates (just track it). If $x_1$ is far away, reaction dominates (teleport the belief).
:::

(pi-wfr-metric)=
::::{admonition} Physics Isomorphism: Wasserstein-Fisher-Rao Geometry
:class: note

**In Physics:** The Wasserstein-Fisher-Rao (WFR) metric on probability measures combines optimal transport (Wasserstein) with information geometry (Fisher-Rao). It is the unique metric allowing both mass transport and creation/annihilation {cite}`liero2018optimal,chizat2018interpolating`.

**In Implementation:** The belief density $\rho$ evolves under the WFR metric on $\mathcal{P}(\mathcal{Z})$:

$$
d_{\text{WFR}}^2(\rho_0, \rho_1) = \inf_{\rho, v, r} \int_0^1 \int_{\mathcal{Z}} \left( \|v\|_G^2 + \lambda^2 r^2 \right) \rho \, d\mu_G \, dt

$$
**Correspondence Table:**
| Optimal Transport | Agent (Belief Dynamics) |
|:------------------|:------------------------|
| Wasserstein distance $W_2$ | Transport cost for belief |
| Fisher-Rao distance | Information cost for reweighting |
| Transport velocity $v$ | Belief flow in $\mathcal{Z}$ |
| Reaction rate $r$ | Mass creation/annihilation |
| Benamou-Brenier formula | Dynamic formulation |
| Geodesic interpolation | Optimal belief transition |

**Significance:** WFR unifies transport (Wasserstein) and reweighting (Fisher-Rao) in a single Riemannian geometry.
::::

:::{prf:remark} Units
:label: rem-units

$[v] = \text{length}/\text{time}$, $[r] = 1/\text{time}$, and $[\lambda] = \text{length}$. The ratio $\|v\|/(\lambda |r|)$ determines whether transport or reaction dominates.

:::

:::{div} feynman-prose
The units tell you something important. Velocity has units of length per time---that's obvious. Reaction rate has units of inverse time---it's a growth rate, like an interest rate. And $\lambda$, the crossover parameter, has units of length.

So when is transport preferred over reaction? When $\|v\|/(\lambda |r|) > 1$, which means when the actual transport velocity is larger than $\lambda$ times the reaction rate. Since $\lambda$ is a length scale, this is saying: if the distance to travel is less than $\lambda$, transport wins; if it's more than $\lambda$, reaction wins.

This is beautiful. The physics tells you exactly when to "teleport" versus when to "walk."
:::

(sec-transport-vs-reaction-components)=
## Transport vs. Reaction Components

:::{div} feynman-prose
Now let's look at the two mechanisms separately before understanding how they combine.
:::

The belief state $\rho_s$ evolves on the bundle $\mathcal{Z}$ via two mechanisms.

**1. Transport (Wasserstein Component):**
The density evolves via the continuity equation $\partial_s\rho + \nabla\cdot(\rho v) = 0$ along the continuous coordinates $z_n$. The transport cost is $\int \|v\|_G^2\, d\rho$. In the limit $r \to 0$, the dynamics reduce to the standard Wasserstein-2 ($W_2$) optimal transport on the Riemannian manifold.

**2. Reaction (Fisher-Rao Component):**
The density undergoes local mass creation/annihilation via the source term $\rho r$. This corresponds to discrete chart transitions: mass decreases on Chart A ($r < 0$) and increases on Chart B ($r > 0$). The reaction cost is $\int \lambda^2|r|^2\, d\rho$. In the limit $v \to 0$, the dynamics reduce to the Fisher-Rao metric on the probability simplex $\Delta^{|\mathcal{K}|}$.

:::{div} feynman-prose
Here's a way to think about the difference.

**Transport** is like rearranging furniture in a room. You can slide the couch from here to there, but the couch is still the same couch, and the total amount of furniture is conserved. The cost depends on how far you move things and how heavy they are.

**Reaction** is like a chemical reaction. You put in reactants, you get out products. Mass isn't conserved locally---it appears and disappears. In our case, the "mass" is belief: probability assigned to different hypotheses.

Both mechanisms are doing something profound. Transport handles the question "how does my estimate of *position* change?" Reaction handles the question "how does my estimate of *what situation I'm in* change?" In a rich agent, both happen simultaneously.
:::

:::{admonition} The Two Extreme Cases
:class: feynman-added tip

| Limit | What dominates | Physical picture | Agent behavior |
|-------|---------------|------------------|----------------|
| $r \to 0$ | Transport | Incompressible fluid flow | Smooth tracking within a hypothesis |
| $v \to 0$ | Reaction | Chemical kinetics | Switching between hypotheses |
| General | Both | Compressible reactive flow | Tracking with hypothesis revision |

Most interesting agent behavior lives in the "general" regime. A robot tracking an object while considering alternative interpretations is doing both transport and reaction.
:::

**3. The Coupling Constant $\lambda$ (Reaction-Transport Crossover Scale):**

This parameter defines the characteristic length scale at which transport cost exceeds reaction cost:
- If $\|z_A - z_B\|_G < \lambda$: Transport is preferred (continuous regime)
- If $\|z_A - z_B\|_G > \lambda$: Reaction is preferred (discrete chart transition)

**Operational interpretation:** $\lambda$ is exactly the **radius of the chart overlap region** ({ref}`Section 7.13 <sec-factorized-jump-operators-efficient-chart-transitions>`). Within overlaps, transport is efficient; across non-overlapping regions, reaction dominates.

:::{div} feynman-prose
The parameter $\lambda$ is the key to the whole thing. It answers the question: "How far can you walk before it's cheaper to teleport?"

Think about it this way. If you're in San Francisco and you want to get to Oakland, you might drive across the bridge. But if you want to get to Tokyo, you're going to fly. The crossover distance where flying becomes preferable to driving is analogous to $\lambda$.

In the agent's latent space, $\lambda$ is (roughly) the size of the overlap between neighboring charts. If two hypotheses are "close" in the sense that they share similar predictions, transport between them is cheap. If they're "far"---if they represent totally different interpretations of the situation---reaction is cheaper. You don't smoothly walk from "this is a cup" to "this is a cat"; you teleport.
:::

:::{prf:definition} Canonical length-scale
:label: def-canonical-length-scale

Let $G$ be the latent metric on $\mathcal{Z}$. The canonical choice for $\lambda$ is the **geodesic injectivity radius**:

$$
\lambda := \min_{z \in \mathcal{Z}} \text{inj}_G(z),

$$
where $\text{inj}_G(z)$ is the injectivity radius at $z$ -- the largest $r$ such that the exponential map $\exp_z: T_z\mathcal{Z} \to \mathcal{Z}$ is a diffeomorphism on $B_r(0)$.

*Default value.* If the injectivity radius is unknown or the metric is learned, a practical default is:

$$
\lambda_{\text{default}} = \sqrt{\frac{\text{tr}(G^{-1})}{n}} \approx \text{mean characteristic length of } \mathcal{Z}.

$$
This corresponds to the RMS geodesic step size in an isotropic metric.

*Cross-reference:* The screening length $\ell_{\text{screen}} = 1/\kappa$ from {ref}`Section 24.2 <sec-the-bulk-potential-screened-poisson-equation>` plays an analogous role for temporal horizons; $\lambda$ plays the corresponding role for spatial horizons in the WFR geometry.

:::

:::{div} feynman-prose
Why the injectivity radius? Because that's the scale at which the manifold's topology starts to matter. Within the injectivity radius, the space looks like flat Euclidean space---you can move in any direction without running into weird topological obstacles. Beyond the injectivity radius, paths might wrap around, and the geometry becomes nontrivial.

If you've never encountered the injectivity radius before, here's the intuition. Imagine you're on the surface of a sphere. From any point, you can draw geodesics (great circles) in all directions. How far can you go before some of those geodesics start meeting again? That distance is the injectivity radius. On a sphere of radius $R$, it's $\pi R$---the distance to the antipodal point.

In the agent's latent space, the injectivity radius tells you: how far can you transport mass along smooth geodesics before you have to worry about the discrete structure of the chart atlas?
:::

(sec-reconciling-discrete-and-continuous)=
## Reconciling Discrete and Continuous

:::{div} feynman-prose
Now comes the payoff. The WFR metric doesn't just let us handle discrete and continuous separately---it actually *unifies* them in a way that respects the structure of both.
:::

:::{prf:proposition} Limiting Regimes
:label: prop-limiting-regimes

The WFR metric seamlessly unifies discrete and continuous dynamics:

1. **Continuous Movement (Flow):** When moving within a chart, $r \approx 0$. The dynamics are dominated by $\nabla \cdot (\rho v)$, and the metric reduces to $W_2$ (Wasserstein-2). This recovers the Riemannian manifold structure of the nuisance fibres.

2. **Discrete Movement (Jump):** When the flow reaches a topological obstruction (chart boundary without overlap), transport becomes infinitely expensive. It becomes cheaper to use the source term $r$:
   - $r < 0$ on the old chart (mass destruction)
   - $r > 0$ on the new chart (mass creation)
   This recovers the **Fisher-Rao metric** on the discrete simplex $\Delta^{|\mathcal{K}|}$.

3. **Mixed Regime (Overlap):** In chart overlaps, both $v$ and $r$ are active. The optimal path smoothly interpolates between transport and reaction.

*Proof sketch.* The cone-space representation of WFR (lifting $\rho$ to $(\sqrt{\rho}, \sqrt{\rho} \cdot z)$) shows that the WFR geodesic projects to a $W_2$ geodesic when $r = 0$, and to a Fisher-Rao geodesic when $v = 0$. $\square$

:::

:::{div} feynman-prose
This is really beautiful. The WFR metric is like a universal adapter. When you're doing ordinary tracking, it acts like a Wasserstein metric. When you're doing hypothesis switching, it acts like a Fisher-Rao metric. And when you're doing both---which is most of the time---it finds the optimal blend.

The "cone-space representation" mentioned in the proof is a technical trick that linearizes the problem. Instead of working with densities $\rho$, you work with $\sqrt{\rho}$. This turns the nonlinear WFR geodesic equation into something much more tractable. But the conceptual point stands without the technical details: WFR smoothly interpolates between the two limiting geometries.
:::

:::{admonition} Analogy: Highway vs. Airplane
:class: feynman-added note

Imagine you're in a landscape of cities connected by highways and airports.

- **Transport (Wasserstein):** Driving on highways. You can go anywhere, but it takes time proportional to distance.
- **Reaction (Fisher-Rao):** Flying between airports. Near-instant, but airports are only at discrete locations (the charts).
- **WFR:** Finding the optimal combination. For short trips, drive. For long trips, drive to the nearest airport, fly, then drive from the destination airport.

The length scale $\lambda$ is like the maximum distance where driving is still cheaper than the overhead of flying. And the WFR metric automatically finds the optimal combination for any origin-destination pair.
:::

::::{admonition} Connection to RL #26: Distributional RL as Degenerate WFR Geometry
:class: note
:name: conn-rl-26
**The General Law (Fragile Agent):**
Belief states evolve on $\mathcal{M}^+(\mathcal{Z})$ via **Wasserstein-Fisher-Rao dynamics**:

$$
d^2_{\text{WFR}}(\rho_0, \rho_1) = \inf \int_0^1 \int_{\mathcal{Z}} \left( \|v_s\|_G^2 + \lambda^2 |r_s|^2 \right) d\rho_s\, ds

$$
subject to the unbalanced continuity equation $\partial_s \rho + \nabla \cdot (\rho v) = \rho r$.

**The Degenerate Limit:**
Restrict to value distributions at single states (no spatial transport). Use Euclidean metric ($G \to I$).

**The Special Case (Standard RL):**

$$
Z(s, a) \stackrel{D}{=} R + \gamma Z(S', A'), \quad Q(s,a) = \mathbb{E}[Z(s,a)]

$$
This recovers **Distributional RL**: C51, QR-DQN, IQN {cite}`bellemare2017c51,dabney2018qrdqn`.

**What the generalization offers:**
- **Unified transport-reaction**: WFR handles continuous flow (within charts) and discrete jumps (between charts) in one framework
- **Belief geometry**: The metric on $\mathcal{M}^+(\mathcal{Z})$ respects both $W_2$ (spatial) and Fisher-Rao (probabilistic)
- **Teleportation length**: $\lambda$ determines when transport beats reaction (Proposition {prf:ref}`prop-limiting-regimes`)
- **GKSL embedding**: Quantum-like master equations embed naturally ({ref}`Section 20.5 <sec-connection-to-gksl-master-equation>`)
::::

(sec-connection-to-gksl-master-equation)=
## Connection to GKSL / Master Equation ({ref}`Section 12.5 <sec-optional-operator-valued-belief-updates>`)

:::{div} feynman-prose
Now let me show you a connection that has rigorous mathematical foundations: the WFR framework and the Lindblad master equation are both gradient flows, and in the classical limit they coincide exactly.

You might ask: what does quantum mechanics have to do with our classical agent? The answer is structural. Both frameworks solve the same problem: how to describe dynamics that combine smooth evolution with sudden jumps, while preserving probability. The mathematics turns out to be the same.
:::

The WFR framework connects rigorously to the GKSL (Lindblad) master equation via the **classical limit**. We state this precisely.

:::{prf:theorem} Classical Master Equation as WFR Gradient Flow
:label: thm-classical-master-equation-wfr

Let $\mathcal{K} = \{1, \ldots, K\}$ be a finite state space with transition rates $W_{jk} \geq 0$ (rate of jumping from $k$ to $j$). The classical master equation

$$
\dot{p}_j = \sum_{k} W_{jk} p_k - W_{kj} p_j
$$

is the **gradient flow** of the relative entropy $H(p \| \pi) = \sum_j p_j \log(p_j / \pi_j)$ with respect to a discrete Wasserstein-type metric, where $\pi$ is the stationary distribution satisfying detailed balance {cite}`maas2011gradient,mielke2011gradient,chow2012fokker`.

:::

:::{prf:corollary} GKSL Classical Limit
:label: cor-gksl-classical-limit

When the GKSL density matrix is diagonal, $\varrho = \mathrm{diag}(p_1, \ldots, p_K)$, the GKSL equation reduces to a classical master equation with rates

$$
W_{jk} = \sum_\ell \gamma_\ell |\langle j | L_\ell | k \rangle|^2.
$$

The commutator term $-i[H, \varrho]$ vanishes identically for diagonal states. By Theorem {prf:ref}`thm-classical-master-equation-wfr`, this evolution is a gradient flow in WFR geometry.

:::

**Correspondence Table (Classical Limit):**

| GKSL Component (diagonal $\varrho$)                                                               | WFR Interpretation                                           |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| $-i[H, \varrho]$ (Commutator)                                                                     | **Vanishes** (no off-diagonal elements to rotate)            |
| $\sum_j \gamma_j(L_j \varrho L_j^\dagger - \frac{1}{2}\{L_j^\dagger L_j, \varrho\})$ (Dissipator) | Reaction rate $r$ (jump-induced mass redistribution)         |
| Probability conservation $\sum_k \dot{p}_k = 0$                                                   | Balanced reaction ($\int r \, d\mu = 0$ globally)            |
| Jump operators $L_j$                                                                              | Transition kernels (where mass teleports to)                 |

:::{prf:remark} Full Quantum Case
:label: rem-full-quantum-wfr

For non-diagonal density matrices (quantum coherences), the appropriate geometric structure is the **quantum Wasserstein distance** of Carlen \& Maas {cite}`carlen2014wasserstein,carlen2017gradient`. The GKSL equation is the gradient flow of quantum relative entropy with respect to this metric. This framework handles coherences but is more complex than the classical WFR theory used here.

:::

:::{div} feynman-prose
Let me be precise about what is rigorous and what is not.

**Rigorous:** When beliefs are classical probability distributions (no quantum coherences), the master equation dynamics are *exactly* a gradient flow in a Wasserstein-type metric. This is a theorem, not an analogy. Maas (2011) and Mielke (2011) proved it for discrete state spaces; Chizat et al. (2018) extended it to continuous spaces with the full WFR metric.

**Also rigorous but different:** For full quantum states with coherences, Carlen \& Maas (2014, 2017) constructed a quantum Wasserstein distance. GKSL is a gradient flow there too. But this is a different metric space (density matrices, not probability measures).

**The practical upshot:** If your agent uses classical beliefs (probability distributions over states), WFR geometry is the *correct* geometric structure---not an approximation or analogy. If you want to extend to quantum-like coherent beliefs, the mathematics exists but requires the quantum Wasserstein framework.
:::

:::{admonition} Why This Connection Matters
:class: feynman-added tip

The GKSL/Lindblad structure isn't just mathematical elegance for its own sake. It comes with important guarantees:

1. **Complete positivity:** The evolution preserves valid probability distributions. You never get negative probabilities.

2. **Trace preservation (optional):** If you want total probability conserved, you can enforce it. If you want to allow mass creation/destruction, you can do that too.

3. **Markovianity:** The evolution depends only on the current state, not the entire history. This makes computation tractable.

4. **Composability:** GKSL evolutions compose nicely. Running one evolution after another gives another valid GKSL evolution.

These are exactly the properties you want for belief dynamics in a well-behaved agent.
:::

(sec-the-unified-world-model)=
## The Unified World Model

:::{div} feynman-prose
Now let's see how all this theory translates into something you can actually implement. The payoff is striking: instead of having separate "macro predictor" and "micro dynamics" modules that you somehow have to coordinate, you get a single unified world model.
:::

The WFR formulation enables a **single World Model** that predicts both transport and reaction, eliminating the need for separate "macro predictor" and "micro dynamics" modules.

:::{prf:definition} WFR World Model
:label: def-wfr-world-model

The policy outputs a generalized velocity field $(v, r)$ to minimize the WFR path length to the target distribution (goal).

```python
import torch
import torch.nn as nn
from typing import Tuple

class WFRWorldModel(nn.Module):
    """
    Unified World Model using Unbalanced Optimal Transport dynamics.

    Predicts the 'Generalized Velocity' (v, r) for belief particles.
    No separate 'discrete' and 'continuous' modules.
    """

    def __init__(
        self,
        macro_embed_dim: int,
        nuisance_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        # Input: particle state + action
        # State includes: macro embedding, nuisance coords, mass (weight)
        input_dim = macro_embed_dim + nuisance_dim + 1 + action_dim

        # Single MLP backbone for unified dynamics
        self.dynamics_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Head 1: Transport velocity (Riemannian motion on fibre)
        self.head_v = nn.Linear(hidden_dim, nuisance_dim)

        # Head 2: Reaction rate (Fisher-Rao mass creation/destruction)
        self.head_r = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z_t: torch.Tensor,           # [B, D] latent state (macro_embed + nuisance)
        mass_t: torch.Tensor,        # [B, 1] particle weight (belief mass)
        action_t: torch.Tensor,      # [B, A] action
        dt: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next state via WFR dynamics.

        Returns:
            z_next: [B, D] next latent state
            mass_next: [B, 1] next particle mass
            v_t: [B, nuisance_dim] transport velocity
            r_t: [B, 1] reaction rate
        """
        # Unified prediction
        inp = torch.cat([z_t, mass_t, action_t], dim=-1)
        feat = self.dynamics_net(inp)

        v_t = self.head_v(feat)  # Transport velocity
        r_t = self.head_r(feat)  # Reaction rate (log-growth)

        # Integrate dynamics (Euler step)
        # Position update (Transport): z' = z + v * dt
        z_next = z_t.clone()
        z_next[..., -self.head_v.out_features:] += v_t * dt

        # Mass update (Reaction): m' = m * exp(r * dt)
        # If r > 0: hypothesis gaining probability (jumping in)
        # If r < 0: hypothesis losing probability (jumping out)
        mass_next = mass_t * torch.exp(r_t * dt)

        return z_next, mass_next, v_t, r_t
```

**How this handles the "Jump" seamlessly:**

- **Deep inside a Chart:** Model predicts $r \approx 0$ and $v \neq 0$. Particle moves normally.
- **Approaching a Boundary:** Model sees invalid description (high prediction error). Predicts $r < 0$ for current chart, $r > 0$ for neighboring chart particles.
- **Result:** Probability mass smoothly "tunnels" between charts without hard discrete switching.

:::

:::{div} feynman-prose
Look at how clean this is. The network takes in a state and outputs two things: a velocity $v$ and a reaction rate $r$. Then you integrate forward using simple Euler steps:
- Position updates additively: $z' = z + v \cdot dt$
- Mass updates multiplicatively: $m' = m \cdot \exp(r \cdot dt)$

The multiplicative update for mass is key. If $r > 0$, the mass grows exponentially. If $r < 0$, it decays exponentially. And if $r = 0$, mass is conserved. This is exactly the dynamics you want for belief: probability mass being redistributed among hypotheses.

The beautiful thing is that the network learns *when* to use transport and when to use reaction. Deep inside a chart, where the continuous dynamics are predictable, it learns $r \approx 0$ and uses transport. Near chart boundaries, where prediction error rises, it learns to shed mass ($r < 0$) and create mass elsewhere ($r > 0$). No hard-coded switching logic. No combinatorial explosion of cases. Just smooth, learned dynamics.
:::

:::{admonition} Particle Filter Interpretation
:class: feynman-added note

You can think of this as a kind of **differentiable particle filter**. Traditional particle filters maintain a swarm of particles, each with a weight (probability mass). Particles move according to the dynamics model, and weights are updated by reweighting. Occasionally, low-weight particles are "killed" and high-weight particles are "duplicated" (resampling).

The WFR world model does essentially the same thing, but continuously and differentiably:
- **Particle movement** corresponds to transport velocity $v$
- **Weight updates** correspond to $m' = m \cdot \exp(r \cdot dt)$
- **Resampling** is implicit: particles with $r < 0$ gradually lose weight; particles with $r > 0$ gain weight

The advantage: everything is differentiable, so you can backprop through the dynamics to train end-to-end.
:::

(sec-scale-renormalization)=
## Scale Renormalization (Connection to {ref}`Section 7.12 <sec-stacked-topoencoders-deep-renormalization-group-flow>`)

:::{div} feynman-prose
Now here's something that makes physicists very happy: the WFR framework connects naturally to renormalization group ideas. If you're not familiar with the renormalization group, don't worry---the intuition is actually straightforward.

The idea is that physical systems often have structure at multiple scales. A turbulent fluid has large eddies containing smaller eddies containing even smaller eddies. An image has global composition, mid-level objects, and fine textures. And crucially, the "rules" at different scales might be different.

In our stacked TopoEncoder architecture, each layer corresponds to a different scale. The WFR metric applies at each scale, but with a scale-dependent parameter $\lambda$.
:::

For stacked TopoEncoders ({ref}`Section 7.12 <sec-stacked-topoencoders-deep-renormalization-group-flow>`), the WFR metric applies recursively with **scale-dependent coupling**.

Recall the WFR action:

$$
\mathcal{E} = \int \left( \|v\|_G^2 + \lambda^2 |r|^2 \right) d\rho

$$
For a hierarchy of layers $\ell = 0, \ldots, L$:

:::{prf:definition} Scale-Dependent Teleportation Cost
:label: def-scale-dependent-teleportation-cost

$$
\lambda^{(\ell)} \propto \sigma^{(\ell)} \quad \text{(jump cost scales with residual variance)}

$$
where $\sigma^{(\ell)}$ is the scale factor from Definition {prf:ref}`def-the-rescaling-operator-renormalization`.

**Interpretation:**
- **Layer 0 (Bulk / IR):** High $\lambda^{(0)}$. Jumping is expensive; macro-structure is rigid. Transport dominates.
- **Layer $L$ (Texture / UV):** Low $\lambda^{(L)}$. "Mass" (texture details) can appear/disappear cheaply. Reaction dominates.

**Correspondence with Cosmological Constant:**
In the capacity-constrained metric law ({ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`, Theorem {prf:ref}`thm-capacity-constrained-metric-law`), the term $\Lambda G_{ij}$ plays the role of a baseline curvature. The correspondence is:

$$
\Lambda^{(\ell)} \sim \frac{1}{(\lambda^{(\ell)})^2}

$$
- Bulk (low $\Lambda$): Flat, rigid, transport-dominated
- Boundary (high $\Lambda$): Curved, fluid, reaction-dominated

:::

:::{div} feynman-prose
This is saying something profound about multi-scale representations. At coarse scales (the "IR" or "bulk"), the structure is rigid. The macro-classification of a scene---"this is a kitchen, not a forest"---doesn't change easily. The teleportation cost $\lambda$ is high, so transport dominates. You don't jump between macro-hypotheses without strong evidence.

At fine scales (the "UV" or "boundary"), the structure is fluid. The exact texture of a surface, the precise shade of a color---these can change rapidly without violating any fundamental constraints. The teleportation cost $\lambda$ is low, so reaction dominates. Fine details can pop in and out without affecting the big picture.

This matches intuition about perception. The "gist" of a scene is established quickly and changes slowly. The fine details are filled in later and can be revised easily.
:::

:::{admonition} The Cosmological Constant Analogy
:class: feynman-added note

The correspondence with the cosmological constant $\Lambda$ is more than just an analogy---it reflects deep mathematical structure.

In general relativity, $\Lambda$ sets the baseline curvature of spacetime. Large $\Lambda$ means highly curved, dynamic spacetime. Small $\Lambda$ means nearly flat, rigid spacetime.

In our framework:
- **Small $\Lambda$ (bulk):** The latent space is nearly flat. Geodesics are almost straight lines. Transport is efficient.
- **Large $\Lambda$ (boundary):** The latent space is highly curved. Geodesics bend strongly. The "landscape" is rough, and jumping becomes preferable to navigating the complex terrain.

The formula $\Lambda \sim 1/\lambda^2$ makes this precise: small teleportation length means large effective curvature.
:::

(sec-connection-to-einstein-equations)=
## Connection to Einstein Equations ({ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`)

:::{div} feynman-prose
We've been talking about the geometry of belief space as if it were fixed. But here's the punchline: the geometry itself is determined by the belief dynamics. This is exactly like Einstein's general relativity, where mass tells spacetime how to curve, and curved spacetime tells mass how to move.

In our case: belief flow tells the latent metric how to curve, and the curved latent metric tells belief how to flow. The mathematics is the same.
:::

The WFR dynamics provide the **stress-energy tensor** $T_{ij}$ that drives curvature in Theorem {prf:ref}`thm-capacity-constrained-metric-law`.

:::{prf:theorem} WFR Stress-Energy Tensor; variational form
:label: thm-wfr-stress-energy-tensor-variational-form

Let the WFR action be

$$
\mathcal{S}_{\mathrm{WFR}}
=
\frac12\int_0^T\int_{\mathcal{Z}}
\rho\left(\|v\|_G^2+\lambda^2 r^2\right)\,d\mu_G\,ds,

$$
with continuity equation

$$
\partial_s\rho+\nabla\!\cdot(\rho v)=\rho r.

$$
Define

$$
T_{ij}:=
-\frac{2}{\sqrt{|G|}}\frac{\delta(\sqrt{|G|}\,\mathcal{L}_{\mathrm{WFR}})}{\delta G^{ij}}
\quad\text{(holding }\rho,v,r\text{ fixed).}

$$
Then

$$
T_{ij}=\rho\,v_i v_j + P\,G_{ij},
\qquad
P=\frac12\,\rho\left(\|v\|_G^2+\lambda^2 r^2\right),

$$
which is the perfect-fluid form with reaction contributing an additive pressure term
{math}`P_{\mathrm{react}}=\tfrac12\lambda^2\rho r^2`.

*Proof sketch.* Vary $\mathcal{S}_{\mathrm{WFR}}$ with respect to $G^{ij}$ while holding
$(\rho,v,r)$ fixed. Use $\delta\|v\|_G^2=-v_i v_j\,\delta G^{ij}$ and
$\delta d\mu_G=-\tfrac12 G_{ij}\delta G^{ij}d\mu_G$, then collect terms to match
$\delta\mathcal{S}_{\mathrm{WFR}}=-\tfrac12\int T_{ij}\delta G^{ij}d\mu_G\,ds$.
See {ref}`Appendix C <sec-appendix-c-wfr-stress-energy-tensor>` for the full derivation. $\square$

:::

:::{div} feynman-prose
Let me decode this. The stress-energy tensor $T_{ij}$ measures "how much stuff is here and how fast is it moving." In relativity, it's the source term in Einstein's equations---it tells spacetime how to curve.

The result has the "perfect fluid" form, which is the simplest physically reasonable stress-energy tensor. There's a density times velocity-squared term (kinetic energy) and a pressure term.

The beautiful thing is that the **reaction** contributes to the pressure. When the agent is doing a lot of hypothesis-switching (high $r$), that creates "pressure" in the latent space, which through the Einstein-like equations causes the geometry to curve.

What does this mean in practice? Regions of high belief dynamics---where the agent is uncertain, where hypotheses are competing---become geometrically different from regions of certainty. The metric literally adapts to where the interesting action is happening.
:::

(pi-stress-energy)=
::::{admonition} Physics Isomorphism: Stress-Energy Tensor
:class: note

**In Physics:** The stress-energy tensor $T_{\mu\nu}$ is derived from the variation of the matter action with respect to the metric: $T_{\mu\nu} = -\frac{2}{\sqrt{-g}}\frac{\delta S_M}{\delta g^{\mu\nu}}$ {cite}`wald1984general`.

**In Implementation:** The WFR stress-energy tensor (Theorem {prf:ref}`thm-wfr-stress-energy-tensor-variational-form`) is:

$$
T_{ij} = \rho v_i v_j + \frac{1}{2}\rho\left(\|v\|_G^2 + \lambda^2 r^2\right) G_{ij}

$$
derived from $\delta \mathcal{S}_{\text{WFR}}/\delta G^{ij}$. This has the standard perfect-fluid form with positive pressure $P = \frac{1}{2}\rho(\|v\|_G^2 + \lambda^2 r^2)$.

**Correspondence Table:**

| Field Theory | Agent (WFR) |
|:-------------|:------------|
| Matter density $\rho_m$ | Belief density $\rho$ |
| 4-velocity $u^\mu$ | Transport velocity $v^i$ |
| Pressure $p$ | Reaction pressure $\frac{\lambda^2}{2}\rho r^2$ |
| Rest mass density | WFR kinetic energy $\frac{1}{2}\rho\|v\|_G^2$ |
::::

**Implications:**
1. **High velocity ($v$):** Agent moves fast through a region → $T_{ij}$ large → curvature $R_{ij}$ increases → latent space contracts. This is the **Natural Gradient** effect derived from first principles.

2. **High reaction ($r$):** Agent jumps frequently → $P_{\mathrm{react}}$ increases → capacity stress increases. This triggers the boundary-capacity constraint (Definition {prf:ref}`def-dpi-boundary-capacity-constraint`).

:::{div} feynman-prose
These implications deserve emphasis.

The first one says: if the agent moves quickly through some region of latent space, that region effectively shrinks. This is exactly what the Natural Gradient does in optimization---it warps parameter space so that steps are appropriately sized regardless of the local geometry. But here it emerges from first principles, not as a heuristic.

The second one says: if the agent is doing a lot of hypothesis switching, that creates computational "pressure" that eventually hits capacity limits. You can't infinitely subdivide your hypotheses; there's a cost. And the WFR framework quantifies that cost through the reaction pressure term.
:::

**Consistency with existing losses:**

| Existing Loss                                    | WFR Interpretation                             | Status     |
|--------------------------------------------------|------------------------------------------------|------------|
| $\mathcal{L}_{\mathrm{pred}}$ (Prediction)       | Minimizing transport cost $\lVert v\rVert_G^2$ | Compatible |
| $\mathcal{L}_{\mathrm{closure}}$ (Macro closure) | Penalizing reaction $r$ in macro channel       | Compatible |
| Dissipation (Axiom D)                            | $r < 0$ (entropy production)                   | Compatible |
| Capacity ($I < C$)                               | Metric curves to keep WFR path within budget   | Compatible |

:::

:::{admonition} Why This Matters for Implementation
:class: feynman-added tip

The compatibility table above is not just theory---it tells you something practical. The WFR framework doesn't throw away your existing loss functions; it reinterprets them.

- Your **prediction loss** is already (implicitly) penalizing transport cost. The better your world model predicts, the less "velocity" is needed to correct the belief.

- Your **closure loss** (keeping the macro-channel predictive) is penalizing unnecessary reaction. If you're switching hypotheses when you don't need to, you're paying reaction cost.

- Your **entropy losses** relate to dissipation. The WFR framework makes explicit when entropy production is "good" (exploring) versus "bad" (inefficient switching).

- Your **capacity constraints** relate to the metric adaptation. The latent geometry curves to keep everything within budget.

So adopting WFR isn't a rewrite; it's a unification of things you're probably already doing.
:::

(sec-comparison-sasaki-vs-wfr)=
## Comparison: Sasaki vs. WFR

:::{div} feynman-prose
Let me summarize the comparison between the old approach (Sasaki-like product metrics) and the new approach (WFR). This table tells the whole story.
:::

| Feature                     | Sasaki (Product Metric)          | WFR (Unbalanced Transport)             |
|-----------------------------|----------------------------------|----------------------------------------|
| **State representation**    | Fixed point                      | Probability mass / belief              |
| **Topology changes**        | Manual patching required         | Handled natively via $r$               |
| **Path type**               | "Walk then Jump" (discontinuous) | Smooth interpolation                   |
| **Optimization**            | Combinatorial + Gradient descent | Convex (generalized geodesics)         |
| **Theoretical consistency** | Ad-hoc construction              | Gradient flow of entropy (rigorous)    |
| **Multi-scale**             | Separate metrics per scale       | Unified with scale-dependent $\lambda$ |

:::{div} feynman-prose
Every row in this table represents a significant improvement. Let me highlight the most important ones.

**Optimization**: The Sasaki approach leads to combinatorial explosions. You have to decide: do I stay in this chart or jump to that one? With $K$ charts and $T$ time steps, you have $K^T$ possible sequences to consider. The WFR approach is convex---you're just finding a geodesic in a well-defined metric space.

**Path type**: "Walk then Jump" is what happens when you don't have a principled way to mix discrete and continuous. You walk until you can't anymore, then you jump. But where's the boundary? How do you decide when to jump? With WFR, there's no discontinuity. The path smoothly interpolates between transport-dominated and reaction-dominated regimes.

**Theoretical consistency**: The Sasaki construction was always a hack. You take a metric here, a metric there, multiply them together, and hope for the best. The WFR metric comes from a variational principle---it's the unique metric with certain desirable properties. There's nothing ad-hoc about it.
:::

(sec-implementation-wfr-consistency-loss)=
## Implementation: WFR Consistency Loss

:::{div} feynman-prose
Now let's get concrete about how to train models with this framework. The key idea is a **consistency loss** that penalizes violations of the unbalanced continuity equation.
:::

:::{prf:definition} WFR Consistency Loss / WFRCheck
:label: def-wfr-consistency-loss-wfrcheck

The cone-space representation linearizes WFR locally. From $\partial_s \rho = \rho r - \nabla \cdot (\rho v)$ and $u = \sqrt{\rho}$, we have $\partial_s u = \frac{\rho r - \nabla \cdot (\rho v)}{2\sqrt{\rho}}$. Define the consistency loss:

$$
\mathcal{L}_{\mathrm{WFR}} = \left\| \sqrt{\rho_{t+1}} - \sqrt{\rho_t} - \frac{\Delta t}{2\sqrt{\rho_t}}\left(\rho_t r_t - \nabla \cdot (\rho_t v_t)\right) \right\|_{L^2}^2

$$
This penalizes deviations from the unbalanced continuity equation.

**Practical implementation:**

```python
def compute_wfr_consistency_loss(
    rho_t: torch.Tensor,       # [B, K] belief over charts at time t
    rho_t1: torch.Tensor,      # [B, K] belief over charts at time t+1
    v_t: torch.Tensor,         # [B, K, d_n] transport velocity per chart
    r_t: torch.Tensor,         # [B, K] reaction rate per chart
    dt: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute WFR consistency loss (cone-space formulation).

    Penalizes violation of unbalanced continuity equation.
    """
    sqrt_rho_t = torch.sqrt(rho_t + eps)
    sqrt_rho_t1 = torch.sqrt(rho_t1 + eps)

    # Approximate divergence term (finite difference)
    # In practice, use automatic differentiation if v is differentiable
    div_rho_v = torch.zeros_like(rho_t)  # Placeholder for nabla . (rho v)

    # Predicted change in sqrt(rho) from: d/ds sqrt(rho) = (rho*r - div(rho*v)) / (2*sqrt(rho))
    predicted_delta = (dt / (2 * sqrt_rho_t + eps)) * (rho_t * r_t - div_rho_v)

    # Actual change
    actual_delta = sqrt_rho_t1 - sqrt_rho_t

    # L2 loss
    loss = ((actual_delta - predicted_delta) ** 2).mean()

    return loss
```

:::

:::{div} feynman-prose
Why work with $\sqrt{\rho}$ instead of $\rho$? This is the "cone-space" trick I mentioned earlier. The original unbalanced continuity equation is nonlinear, which makes optimization hard. But if you change variables to $u = \sqrt{\rho}$, the resulting equation is much better behaved.

The consistency loss is simple: you predict what $\sqrt{\rho_{t+1}}$ should be based on the current state and the $(v, r)$ outputs, then you penalize the squared difference from the actual $\sqrt{\rho_{t+1}}$.

The divergence term $\nabla \cdot (\rho v)$ is the trickiest part computationally. In practice, you can approximate it with finite differences, or use automatic differentiation if your velocity field is differentiable.
:::

:::{admonition} Implementation Notes
:class: feynman-added note

A few practical considerations:

1. **The $\epsilon$ stabilizer:** We add $\epsilon = 10^{-6}$ inside the square root to avoid division by zero when $\rho \approx 0$. This corresponds to a tiny uniform "background" belief.

2. **The divergence term:** The placeholder `div_rho_v = torch.zeros_like(rho_t)` in the code is a simplification. For a full implementation, you'd need to either:
   - Discretize the divergence using finite differences on a grid
   - Use automatic differentiation through a neural velocity field
   - Use a divergence-free parameterization and ignore this term

3. **Batching:** The loss is computed per-batch and averaged. You want to see it decrease during training, indicating that the world model's $(v, r)$ predictions are becoming more consistent with actual belief evolution.

4. **Scaling:** The loss magnitude depends on $dt$ and the scale of $\rho$. You may need to tune the loss weight relative to other training objectives.
:::

(sec-node-wfrcheck)=
## Node 23: WFRCheck

:::{div} feynman-prose
Finally, we define a diagnostic node that monitors WFR consistency at runtime. This fits into the larger diagnostic framework described in Section 3.
:::

Following the diagnostic node convention ({ref}`Section 3.1 <sec-theory-thin-interfaces>`), we define:

| **#**  | **Name**     | **Component**   | **Type**                 | **Interpretation**          | **Proxy**                    | **Cost** |
|--------|--------------|-----------------|--------------------------|-----------------------------|------------------------------|----------|
| **23** | **WFRCheck** | **World Model** | **Dynamics Consistency** | Transport-Reaction balance? | $\mathcal{L}_{\mathrm{WFR}}$ | $O(BK)$  |

**Trigger conditions:**
- High $\mathcal{L}_{\mathrm{WFR}}$: World model's $(v, r)$ predictions violate continuity
- Remedy: Increase training on transitions; check for distribution shift

:::{div} feynman-prose
When should you worry about this diagnostic? A high WFRCheck loss means your world model is predicting belief dynamics that don't satisfy the continuity equation. There are two common causes:

1. **Insufficient training:** The model simply hasn't learned the dynamics well enough. Solution: more training data, more model capacity, or longer training.

2. **Distribution shift:** The environment has changed in a way the model wasn't trained for. The model is applying its learned $(v, r)$ predictions to situations where they don't apply. Solution: detect the shift and trigger adaptation or re-training.

In either case, high WFRCheck is a warning sign that belief updates may be inconsistent or erratic.
:::

:::{admonition} Summary: What WFR Buys You
:class: feynman-added tip

Let me summarize the key benefits of the WFR framework:

1. **Unified treatment of discrete and continuous:** No more separate modules, no more ad-hoc switching logic.

2. **Principled cost for "jumps":** The parameter $\lambda$ determines when transport beats reaction, derived from the geometry rather than chosen arbitrarily.

3. **Convex optimization:** Finding optimal belief trajectories is a convex problem, not a combinatorial nightmare.

4. **Connects to physics:** The same mathematical structures that describe fluid dynamics, thermodynamics, and general relativity describe belief dynamics. This isn't coincidence---it reflects deep structure.

5. **Enables diagnosis:** The WFRCheck loss gives you a principled way to monitor whether your world model is behaving consistently.

The framework is mathematically sophisticated, but the core intuition is simple: treat belief as a fluid that can flow and react, measure the cost of both, and find the cheapest path.
:::
