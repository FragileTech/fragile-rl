(sec-ontological-expansion-topological-fission-and-the-semantic-vacuum)=
# Ontological Expansion: Topological Fission and the Semantic Vacuum

## TLDR

- Formalize **concept creation** as a dynamical instability: when residual structure cannot be represented, the latent
  atlas undergoes a bifurcation (topological fission).
- The “semantic vacuum” at $z=0$ is the symmetric reference state; under stress, symmetry breaks and new charts/experts
  emerge.
- This chapter turns ontology growth into something monitorable: define stress signals and thresholds for when to spawn
  new representational structure.
- The goal is controlled growth: expand capacity when needed, but avoid runaway fragmentation.
- Connects directly to the capacity-constrained metric law and the metabolism chapter (growth has thermodynamic cost).

## Roadmap

1. Define the semantic vacuum and the notion of ontological stress.
2. Derive topological fission as a pitchfork-like bifurcation.
3. Practical triggers, diagnostics, and how ontology growth interacts with stability/metabolism.

:::{div} feynman-prose
Let me tell you about one of the most interesting problems in building intelligent agents: how do you learn new concepts?

Not just new facts, mind you. Not just "the capital of France is Paris." I mean genuinely new *categories*---new ways of carving up the world. A child who has only ever seen dogs and cats, and then encounters a horse for the first time. An AI trained on chess that suddenly needs to understand poker. A scientist looking at data that doesn't fit any existing theory.

The standard approach in machine learning is to decide upfront how many categories you need, bake that into your architecture, and hope for the best. If you guessed wrong---if the world is more complex than your architecture can represent---too bad. Your model will fail in ways that are hard to diagnose and impossible to fix without retraining from scratch.

That's absurd, isn't it? Humans don't work that way. We encounter novel situations, recognize when our existing categories are inadequate, and create new ones. That's what we're going to formalize here: the mechanism by which an agent expands its ontology---its system of concepts---when the existing structure proves insufficient.

The key insight is that this process has a precise geometric description. It's a *bifurcation*---a phase transition in the space of possible representations. And like any good phase transition, it has a critical point: enough stress in the system, and the old equilibrium becomes unstable. New structure spontaneously emerges.
:::

This section formalizes the mechanism by which agents expand their ontology---creating new conceptual distinctions---when the existing chart structure proves insufficient. The central object is the **Semantic Vacuum** at the origin $z=0$, where the agent's representation is maximally uncertain. Under **Ontological Stress**, this vacuum becomes unstable and undergoes **Topological Fission**: a supercritical pitchfork bifurcation (Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`) that spawns new chart queries.

(rb-dynamic-architecture)=
:::{admonition} Researcher Bridge: Dynamic Architecture vs. Fixed Capacity
:class: tip
Standard models have fixed tensor shapes chosen at initialization. If the environment's complexity exceeds the model's capacity, it fails. **Ontological Fission** is our version of "Dynamic Architecture Growth." When the agent detects "Ontological Stress" (unaccounted-for structure in the noise floor), it triggers a **pitchfork bifurcation** to spawn new latent charts (experts). The model grows to match the data, rather than trying to cram the world into a fixed bottleneck.
:::

**Abstract.** We formalize the expansion of the latent manifold $(\mathcal{Z}, G)$ under representational stress. The
**Semantic Vacuum** $\emptyset$ is defined as the fiber over the origin ($z=0$), characterized by maximal $SO(D)$
symmetry. When the residual texture $z_{\mathrm{tex}}$ exhibits temporal predictability---violating
**Bulk-Boundary Decoupling** (Axiom {prf:ref}`ax-bulk-boundary-decoupling`)---the manifold undergoes **Topological
Fission**: a supercritical pitchfork bifurcation that instantiates new chart queries, expanding the agent's categorical
structure.

*Cross-references:*
- {ref}`sec-radial-generation-entropic-drift-and-policy-control` (Pitchfork bifurcation, $SO(D)$ symmetry).
- {ref}`sec-tier-the-attentive-atlas` (Attentive Atlas, chart queries).
- {ref}`sec-main-result` (Capacity-constrained metric).
- {ref}`sec-variance-value-duality-and-information-conservation` (Entropy-regularized objective).

*Literature:* Symmetry breaking in dynamical systems {cite}`strogatz2015nonlinear`; Ricci flow
{cite}`hamilton1982ricci,perelman2002entropy`; ontology learning {cite}`wong2012`.



(sec-the-semantic-vacuum-as-a-reference-measure)=
## The Semantic Vacuum as a Reference Measure

:::{div} feynman-prose
Here's a question: what does it look like when an agent knows *nothing*?

Not nothing about some particular topic. I mean genuinely maximum ignorance---a state where all possibilities are equally likely, where no direction is preferred over any other. This turns out to be a very special state, and it sits right at the center of our representation space.

Think about it this way. If I asked you "what animal is this?" and showed you a blank screen, you'd have to say "I have no idea---could be anything." That's maximum entropy, maximum uncertainty. And in our geometric picture, that state of maximum uncertainty corresponds to the origin---the center of the Poincare disk.

Why the center? Because of symmetry. At the center, there's nothing to distinguish one direction from another. Move a little bit in the $x$ direction, that's the same as moving in the $y$ direction, or any other direction. The center is the unique point with full rotational symmetry.

We call this state the **Semantic Vacuum**. It's not a vacuum in the sense of "nothing there"---it's a vacuum in the sense of "no information yet." It's the blank slate from which all concepts emerge.

And here's the crucial thing: this vacuum is *unstable*. You can't stay there. Any tiny perturbation---any hint of structure in the data---will push you away from the center and toward some more specific representation. That instability is precisely what enables learning.
:::

At the origin of the Poincare disk, the agent's belief state is maximally uncertain---all directions are equally probable. This is the **Semantic Vacuum**: the unique fiber over $z=0$ in the latent bundle.

:::{prf:definition} Semantic Vacuum
:label: def-semantic-vacuum

Let $(\mathbb{D}, G)$ be the Poincare disk with metric $G_{ij}(z) = 4\delta_{ij}/(1-|z|^2)^2$ (Definition {prf:ref}`def-hyperbolic-volume-growth`). The **Semantic Vacuum** is the fiber

$$
\emptyset := \{z \in \mathcal{Z} : |z| = 0\} = \{0\} \times \mathcal{Z}_{\text{tex}},

$$
equipped with the following properties:

1. **$SO(D)$ Symmetry:** At $z=0$, the metric is isotropic $G(0) = 4I$ (Proposition {prf:ref}`prop-so-d-symmetry-at-origin`), and the entropic force vanishes: $F_{\text{entropy}}(0) = 0$. The system has full rotational symmetry $SO(D)$.

2. **Infrared Limit:** For any TopoEncoder scale $\tau$ ({ref}`sec-rigorous-interpretation-renormalization-group-flow`), $\lim_{\tau \to 0} z(\tau) = \emptyset$. The vacuum is the coarsest resolution.

3. **Reference Measure:** The vacuum carries the Dirac reference measure $\delta_0$ on the bulk coordinates $(K, z_n)$:

   $$
   \mu_{\emptyset} := \delta_0 \otimes \mathcal{N}(0, \sigma_{\text{tex}}^2 I),

   $$
   where the texture component is drawn from the isotropic prior (Definition {prf:ref}`def-boundary-texture-distribution` with $G^{-1}(0) = I/4$).

4. **Information Content:** At the vacuum, $U(0) = 0$ (Definition {prf:ref}`prop-isotropic-radial-expansion`), corresponding to zero information content (maximum entropy).

*Units:* $[\mu_{\emptyset}]$ is a probability measure; $[U] = \mathrm{nat}$.

*Remark (Unstable Origin).* The vacuum is not a fixed point of the radial dynamics: the entropic drift $(1-r^2)/2 > 0$ at $r=0$ implies trajectories immediately expand outward (Theorem {prf:ref}`thm-angular-symmetry-breaking`). The $SO(D)$ angular symmetry is broken by the policy or thermal fluctuations during this expansion.

:::

:::{div} feynman-prose
Now, here's an important practical question: how does the agent actually *get* to this vacuum state? Under what circumstances does the routing system send observations to the center?

The answer is beautifully symmetric: when an observation is equally compatible with *all* existing categories---or equally *incompatible* with all of them. If the observation matches every chart equally well (or equally poorly), the router has no basis for preferring one over another, so it assigns uniform weights. And if the codebooks are centered (which we enforce as an architectural requirement), that uniform weighting maps straight to the origin.

This is exactly what should happen! An observation that doesn't fit any existing category should be recognized as "genuinely novel"---and that recognition is represented by being placed at the semantic vacuum.
:::

:::{prf:lemma} Default Mapping to Vacuum
:label: lem-default-mapping-to-vacuum

Let $\{q_i\}_{i=1}^{N_c}$ be the chart query bank (Definition {prf:ref}`def-attentive-routing-law`) and assume the queries are **centered**: $\sum_{i=1}^{N_c} q_i = 0$. Then for any key $k(x)$ such that all inner products are equal---$\langle q_i, k(x) \rangle = c$ for all $i$---the router weights are uniform:

$$
w_i(x) = \frac{1}{N_c} \quad \forall i \in \{1, \ldots, N_c\}.

$$
The resulting soft codebook embedding is the **barycenter**:

$$
z_q(x) = \sum_{i=1}^{N_c} w_i(x) e_{i, K_{\text{code},i}(x)} = \frac{1}{N_c} \sum_{i=1}^{N_c} e_{i,*},

$$
which equals $0$ if the per-chart codebooks are also centered ($\sum_c e_{i,c} = 0$ for each chart $i$).

*Proof.* From Definition {prf:ref}`def-attentive-routing-law`, $w_i(x) = \exp(\langle q_i, k(x)\rangle/\sqrt{d}) / \sum_j \exp(\langle q_j, k(x)\rangle/\sqrt{d})$. If $\langle q_i, k(x)\rangle = c$ for all $i$, then $w_i = e^{c/\sqrt{d}} / (N_c \cdot e^{c/\sqrt{d}}) = 1/N_c$. The soft code $z_q$ is the weighted sum; under centering, this is the barycenter at $0$. $\square$

*Interpretation.* When the observation $x$ is equally compatible with all charts (or incompatible with all), the router outputs uniform weights. Under centering, this maps to the vacuum---the maximum-entropy state in latent space.

**Architectural Requirement 30.1.3 (Codebook Centering).** To ensure the vacuum is reachable, initialize and regularize codebooks to satisfy $\sum_i q_i = 0$ and $\sum_c e_{i,c} = 0$. This can be enforced via:

$$
\mathcal{L}_{\text{center}} := \left\|\sum_{i=1}^{N_c} q_i\right\|^2 + \sum_{i=1}^{N_c} \left\|\sum_{c=1}^{N_v} e_{i,c}\right\|^2.

$$
:::



(sec-ontological-stress)=
## Ontological Stress

:::{div} feynman-prose
Now we come to the central diagnostic: how does an agent know when its ontology is inadequate?

The answer is subtle and beautiful. Remember the "texture" component $z_{\text{tex}}$? This is supposed to be the *leftover*---everything in the observation that couldn't be captured by the discrete category $K$ and the continuous nuisance $z_n$. It's the reconstruction residual, the noise floor, the stuff we couldn't compress.

And here's the key: if our ontology is adequate, this residual should be *unpredictable*. It should be white noise. Knowing the texture at time $t$ should tell you nothing about the texture at time $t+1$.

Why? Because if the texture *is* predictable, that means there's structure in it---structure that should have been captured by our macro-state but wasn't. It's information hiding in what we claimed was noise.

Think about it this way. Suppose you're classifying animals, and your only categories are "has fur" and "doesn't have fur." You observe a sequence of animals and notice that within the "has fur" category, the texture residuals are correlated over time---furry animals tend to be followed by other furry animals with similar textures. That correlation is a signal! It means there's a distinction you're not capturing. Maybe "dog" vs. "cat" vs. "bear." Your ontology is too coarse.

We call this predictability in the texture channel **Ontological Stress**. It's a measure of how much structure we're missing---how much our current concepts fail to carve nature at its joints.
:::

The existing chart structure may be insufficient to discriminate observations that differ in task-relevant ways. We quantify this **Ontological Stress** via the conditional mutual information between consecutive texture components.

:::{prf:definition} Ontological Stress
:label: def-ontological-stress

Let $(K_t, z_{n,t}, z_{\text{tex},t})$ be the agent's state at time $t$ (Definition {prf:ref}`def-bounded-rationality-controller`). The **Ontological Stress** is the conditional mutual information:

$$
\Xi := I(z_{\text{tex},t}; z_{\text{tex},t+1} \mid K_t, z_{n,t}, K^{\text{act}}_t),

$$
where $I(\cdot;\cdot|\cdot)$ denotes conditional mutual information in nats.

*Units:* $[\Xi] = \mathrm{nat}$ (dimensionless information).

*Interpretation.* By Axiom {prf:ref}`ax-bulk-boundary-decoupling` (Bulk-Boundary Decoupling), texture should be unpredictable -- a white-noise residual. If $\Xi > 0$, then texture at time $t$ predicts texture at time $t+1$, conditional on the macro-state and action. This violates the partition condition: the texture channel contains structure that should have been captured by $(K, z_n)$ but was not. The agent's ontology is **too coarse**.

*Cross-reference.* Compare with the closure defect $I(K_{t+1}; Z_t \mid K_t, K^{\text{act}}_t)$ ({ref}`sec-conditional-independence-and-sufficiency`). Ontological Stress is the dual: predictability *within* texture rather than *from* texture to macro.

:::

:::{admonition} Example: The Hidden Structure
:class: feynman-added example

Here's a concrete example. Suppose you're building a trading agent with two market regimes: "trending" and "mean-reverting." You've encoded this as your discrete state $K \in \{\text{trend}, \text{mean-revert}\}$.

Now suppose there's actually a third regime---"choppy sideways"---that you haven't identified. What happens?

Your encoder will classify choppy-sideways observations as either trending or mean-reverting (probably randomly). The macro-state $K$ will be wrong, but more tellingly: the *texture residual* will be predictable. During choppy-sideways periods, the residual at time $t$ will look similar to the residual at time $t+1$, because the same unmodeled structure is being pushed into the noise floor.

The ontological stress $\Xi$ will spike. It's the system detecting: "There's a pattern here I'm not capturing."
:::

:::{prf:theorem} Vacuum Concentration Under Unknown Unknowns
:label: thm-vacuum-concentration-under-unknown-unknowns

Let $\mathcal{F}[p, \pi]$ be the entropy-regularized objective (Definition {prf:ref}`def-entropy-regularized-objective-functional`):

$$
\mathcal{F}[p, \pi] = \int_{\mathcal{Z}} p(z) \Big( V(z) - \tau H(\pi(\cdot|z)) \Big) d\mu_G.

$$
If the value function $V$ is **uninformative** in a region $\Omega \subset \mathcal{Z}$ -- i.e., $\nabla_A V|_\Omega \approx 0$ and $\nabla^2 V|_\Omega \approx 0$ -- then the entropy term dominates and the optimal belief concentrates toward maximum-entropy configurations:

$$
p^*(z) \propto \exp\left(-\frac{V(z)}{\tau}\right) \xrightarrow{\nabla_A V \to 0} \text{uniform on } \Omega.

$$
In the Poincare disk geometry, the maximum-entropy state is the vacuum $z = 0$.

*Proof sketch.* The stationary distribution of the Langevin dynamics (Definition {prf:ref}`def-bulk-drift-continuous-flow`) is $p(z) \propto \exp(-\Phi_{\text{eff}}(z)/T_c)$ where $\Phi_{\text{eff}}$ includes the hyperbolic potential $U(z)$. When $V$ is flat, $\Phi_{\text{eff}} \approx U(z) = -2\operatorname{artanh}(|z|)$, which is maximized at $z = 0$. The entropic drift $-\nabla_G U$ vanishes at the origin (Proposition {prf:ref}`def-hyperbolic-information-potential`), making it the unique stationary point. $\square$

*Interpretation.* When encountering observations outside the learned structure, the MaxEnt policy concentrates at the vacuum, correctly representing maximum uncertainty.

*Remark (Capacity Tension).* If belief mass accumulates at the vacuum such that bulk information $I_{\mathrm{bulk}}$ approaches the boundary capacity $C_\partial$ (the Capacity-Constrained Metric Law, Theorem {prf:ref}`thm-capacity-constrained-metric-law`), the current chart structure is insufficient. This tension -- high information density at a single point -- indicates fission is required to distribute the representational load.

:::

:::{div} feynman-prose
This theorem tells us something profound: when the agent encounters genuine novelty---observations it can't make sense of---it automatically concentrates at the vacuum. This isn't a failure mode; it's the correct behavior. The agent is saying, "I don't know what category this belongs to. All my existing concepts are equally (ir)relevant."

But here's the problem: you can't stay at the vacuum forever. If lots of observations are accumulating there, you're losing information. You're compressing genuinely distinct things into a single "I don't know" bucket. The system is under *capacity pressure*, and something has to give.

What gives is the topology itself. When enough stress accumulates, the vacuum becomes unstable and *splits*---spawning new concepts to accommodate the structure that was hiding in the noise.
:::

:::{admonition} Connection to RL #11: RND as Degenerate Ontological Stress
:class: note
:name: conn-rl-11
**The General Law (Fragile Agent):**
**Ontological Stress** measures predictability in the texture channel:

$$
\Xi := I(z_{\text{tex},t}; z_{\text{tex},t+1} \mid K_t, z_{n,t}, K^{\text{act}}_t)

$$
When $\Xi > \Xi_{\text{crit}}$, the system triggers **topological fission**: a pitchfork bifurcation that expands the chart structure ({ref}`sec-symmetry-breaking-and-chart-birth`).

**The Degenerate Limit:**
Set $\Xi_{\text{crit}} \to \infty$ (never fission). Instead, feed $\Xi$ directly into the reward function as an exploration bonus.

**The Special Case (Standard RL):**

$$
r_{\text{RND}} = r + \beta \cdot \|f(s) - \hat{f}(s)\|^2

$$
This recovers **Random Network Distillation (RND)**---prediction error as "curiosity" reward.

**Result:** RND agents get "high" on Ontological Stress but never fix the underlying problem. They explore novel states but don't expand their representational capacity to *understand* them. The Fragile Agent uses $\Xi$ as a diagnostic trigger, not a reward signal.

**What the generalization offers:**
- Structural response: high $\Xi$ triggers chart fission, expanding ontology
- Principled threshold: $\Xi_{\text{crit}}$ balances exploration cost vs. complexity cost
- No reward hacking: exploration is architectural, not incentive-based
:::



(sec-the-fission-criterion)=
## The Fission Criterion

:::{div} feynman-prose
So we've established that ontological stress signals a need for new concepts. But should we always respond by creating them?

Absolutely not. Every new concept has a cost. More parameters to store. More computation to route. More ways for the system to overfit. If I created a new category every time I saw something slightly unusual, I'd end up with a million categories and zero generalization.

This is the bias-variance tradeoff in a new guise. Too few concepts, and you're underfitting---lumping together things that should be distinguished. Too many concepts, and you're overfitting---distinguishing things that don't matter.

The right answer is a *threshold*: create new concepts only when the stress is high enough, and only when the expected benefit (better value estimates, better predictions) exceeds the cost (more complexity, more parameters).

This is Occam's Razor, formalized. Expand your ontology only when the data *demands* it.
:::

Not all ontological stress justifies expansion. Creating new charts incurs **complexity costs** (additional parameters, increased inference time). We formalize when expansion is warranted.

:::{prf:axiom} Ontological Expansion Principle
:label: ax-ontological-expansion-principle

The agent should expand its chart structure (increase $N_c$) if and only if the expected value improvement exceeds the complexity cost:

$$
\mathbb{E}\left[\Delta V \mid \text{fission}\right] > \mathcal{C}_{\text{complexity}}(N_c \to N_c + 1),

$$
where $\Delta V$ is the value gain from finer discrimination and $\mathcal{C}_{\text{complexity}}$ is measured in nats (to match units with value).

*Remark.* This is the MDL/rate-distortion principle ({ref}`sec-the-shutter-as-a-vq-vae`) applied to ontology: expand only if the distortion reduction exceeds the rate increase.

:::
:::{prf:theorem} Fission Criterion
:label: thm-fission-criterion

Let $\Xi$ be the Ontological Stress (Definition {prf:ref}`def-ontological-stress`) and let $\Xi_{\text{crit}} > 0$ be a threshold. Let $\Delta V_{\text{proj}}$ be the projected value improvement from splitting the highest-stress chart. The fission criterion is:

$$
\text{Fission} \iff \Xi > \Xi_{\text{crit}} \quad \text{AND} \quad \Delta V_{\text{proj}} > \mathcal{C}_{\text{complexity}}.

$$
*Units:* All quantities are in nats. The complexity cost $\mathcal{C}_{\text{complexity}}(N_c \to N_c + 1)$ includes the entropy increase $\log((N_c+1)/N_c)$ from the expanded codebook plus any regularization penalty on parameter count.

:::

:::{div} feynman-prose
Notice this is a two-part criterion. You need *both* high stress (there's structure you're missing) *and* positive expected value improvement (that structure actually matters for your objectives). You might have high stress on something completely irrelevant to your task---in that case, don't expand. Let it stay in the noise floor where it belongs.
:::

:::{admonition} Connection to RL #12: Fixed Architecture as Degenerate Fission
:class: note
:name: conn-rl-12
**The General Law (Fragile Agent):**
When Ontological Stress $\Xi > \Xi_{\text{crit}}$, the system triggers a **pitchfork bifurcation** that spawns new chart queries:

$$
\frac{dr}{ds} = (\Xi - \Xi_{\text{crit}}) r - \alpha r^3 + \sigma\xi

$$
The network topology **expands** to accommodate new conceptual distinctions.

**The Degenerate Limit:**
Set $\Xi_{\text{crit}} \to \infty$ (infinite fission threshold). The network never expands.

**The Special Case (Standard Deep Learning):**

$$
|\theta| = \text{const} \quad \text{(parameter count fixed at initialization)}

$$
This recovers **standard deep learning**---the agent can never learn a concept that doesn't fit in its initial tensor shapes.

**Result:** Standard networks face a capacity crisis: they must either (1) compress concepts beyond recognition, or (2) fail silently when encountering structure beyond their representational budget. The Fragile Agent grows its ontology to match task complexity.

**What the generalization offers:**
- Dynamic capacity: network topology adapts to task requirements
- Principled growth: fission occurs when complexity cost is justified by value gain
- No catastrophic forgetting: new charts are topologically isolated ({ref}`sec-summary-the-lifecycle-of-an-ontology`)
:::



(sec-symmetry-breaking-and-chart-birth)=
## Symmetry Breaking and Chart Birth

:::{div} feynman-prose
Now for the main event: how does a new concept actually come into being?

The answer is a *symmetry breaking*. Before fission, there's one query vector $q_i$ representing a single category. After fission, there are two: $q_i^+$ and $q_i^-$, pointing in slightly different directions. The single concept has split into two.

This is exactly like a pitchfork bifurcation in dynamical systems. Think of a ball sitting at the top of a hill. If the hill is stable (concave down), the ball stays put. But if you gradually flatten the hill and then invert it (make it concave up), the ball will roll off---and it has to choose a direction. Left or right. The original symmetric state becomes unstable, and the system spontaneously breaks symmetry by picking one of two equivalent alternatives.

In our case, the "ball" is the position of the chart query, and the "hill" is the loss landscape. When ontological stress exceeds the critical threshold, the original query position becomes unstable. The query "rolls off the hill" in some direction $u$, and what was one query becomes two: $q_i \pm \epsilon u$.

But here's the elegant part: both daughters are equivalent by symmetry. There's no intrinsic difference between $q_i + \epsilon u$ and $q_i - \epsilon u$. They're mirror images. The asymmetry---which daughter gets which observations---emerges from the data, not from any bias in the algorithm.

This is how new concepts are born: not by design, but by instability. When the pressure gets high enough, the old structure cracks and new structure emerges.
:::

When the Fission Criterion is satisfied, the agent creates a new chart by splitting an existing query vector. This process is a **supercritical pitchfork bifurcation** in the space of chart queries (Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`).

:::{prf:definition} Query Fission
:label: def-query-fission

Let $q_i \in \mathbb{R}^d$ be a chart query vector ({ref}`sec-tier-the-attentive-atlas`) with associated codebook $\{e_{i,c}\}_{c=1}^{N_v}$. A **query fission** replaces $q_i$ with two daughter queries:

$$
q_i \mapsto \{q_i^+, q_i^-\} := \{q_i + \epsilon u, q_i - \epsilon u\},

$$
where $u \in \mathbb{R}^d$ is the **fission direction** (unit vector) and $\epsilon > 0$ is the **fission amplitude**.

The daughter codebooks are initialized as copies:

$$
e_{i^\pm, c} := e_{i, c} \quad \forall c \in \{1, \ldots, N_v\}.

$$
*Selection of fission direction.* The optimal $u$ maximizes the variance of router assignments under the new queries:

$$
u^* = \arg\max_{\|u\|=1} \text{Var}_{x \sim \mathcal{D}}\left[\langle k(x), u \rangle \mid w_i(x) > 1/N_c\right],

$$
i.e., the principal component of keys within the chart's Voronoi cell.

:::

:::{div} feynman-prose
The fission direction $u$ is chosen to maximize *discriminability*. We're asking: "Within this chart's current territory, what's the axis of maximum variation?" That axis becomes the fission direction. One daughter will handle observations on one side of this axis; the other daughter handles the other side.

This is just PCA within a partition---finding the direction of greatest spread among the observations currently assigned to this chart, and splitting along that direction.
:::

:::{prf:theorem} Supercritical Pitchfork Bifurcation for Charts {cite}`strogatz2015nonlinear`
:label: thm-supercritical-pitchfork-bifurcation-for-charts

The query fission dynamics exhibit a **supercritical pitchfork bifurcation**. Let $r := \|q_i^+ - q_i^-\|/2 = \epsilon$ be the half-separation of daughter queries. The radial evolution satisfies:

$$
\frac{dr}{ds} = (\Xi - \Xi_{\text{crit}}) r - \alpha r^3 + \sigma\xi,

$$
where:
- $\Xi - \Xi_{\text{crit}}$ is the **bifurcation parameter** (supercritical when positive)
- $\alpha > 0$ is a stabilizing cubic coefficient (from competition for training data)
- $\sigma\xi$ is noise from stochastic gradient updates
- $s$ is the training step (flow time)

**Phase Transition:**
1. **Sub-critical ($\Xi < \Xi_{\text{crit}}$):** $r=0$ is the unique stable fixed point. The daughters collapse back to the parent ($r \to 0$).
2. **Super-critical ($\Xi > \Xi_{\text{crit}}$):** $r=0$ becomes unstable. The daughters separate toward a new equilibrium:

   $$
   r^* = \sqrt{\frac{\Xi - \Xi_{\text{crit}}}{\alpha}}.

   $$
*Proof.* The dynamics derive from the effective potential:

$$
\Phi_{\text{fission}}(r) = -\frac{(\Xi - \Xi_{\text{crit}})}{2} r^2 + \frac{\alpha}{4} r^4,

$$
which has the standard pitchfork normal form. For $\Xi > \Xi_{\text{crit}}$, the origin has $\Phi_{\text{fission}}''(0) = -(\Xi - \Xi_{\text{crit}}) < 0$, becoming unstable. Stable minima appear at $r = \pm r^*$. The cubic term arises from router saturation: as daughters separate, they compete for data, and the loss landscape penalizes excessive separation. $\square$

*Critical Temperature Constraint.* The barrier height of the effective potential is $\Delta\Phi = (\Xi - \Xi_{\text{crit}})^2 / (4\alpha)$. Thermal fluctuations can restore symmetry (collapse daughters) if cognitive temperature ({prf:ref}`def-cognitive-temperature`) exceeds this barrier. For stable fission, require:

$$
T_c < \frac{(\Xi - \Xi_{\text{crit}})^2}{4\alpha}.

$$
:::

:::{admonition} The Physics of Concept Birth
:class: feynman-added note

Let me give you the physical picture. The equilibrium separation $r^* = \sqrt{(\Xi - \Xi_{\text{crit}})/\alpha}$ tells you how far apart the daughter concepts end up.

Just above the critical stress ($\Xi \approx \Xi_{\text{crit}}$), the separation is tiny: the concepts are almost indistinguishable. As stress increases, the separation grows like a square root---slowly at first, then faster.

The cubic term $-\alpha r^3$ is what prevents runaway separation. It comes from data competition: as the daughters get further apart, they're fighting over fewer shared observations, and the training signal weakens. Eventually they reach an equilibrium where the expansive force (from stress) balances the contractive force (from data competition).

And there's a temperature condition! If the "cognitive temperature" is too high---if the system is too noisy, too exploratory, too jittery---the thermal fluctuations can collapse the daughters back together. The fission only sticks if the system is cool enough to maintain the separation.
:::



(sec-metric-relaxation-ontological-ricci-flow)=
## Metric Relaxation: Ontological Ricci Flow

:::{div} feynman-prose
After a fission event, the geometry of the latent space needs to relax. The metric tensor $G$---which tells us how to measure distances in the space---was calibrated for the *old* chart structure. Now there are new charts, and the metric needs to adapt.

This is where something beautiful happens. The adaptation process follows a *Ricci flow*---the same geometric evolution that Perelman used to prove the Poincare conjecture (one of the most famous results in 21st-century mathematics).

The basic idea is simple: curvature should flow toward uniformity. Regions with high curvature should "spread out," and regions with low curvature should "bunch up," until everything equilibrates. In our context, this means the metric adapts to give each chart an appropriate "territory" in the latent space.

But we add one extra term: the Hessian of ontological stress. This term creates curvature precisely where new distinctions are needed. High stress gradient means "we need more resolution here," and the metric responds by expanding in that direction.

This is geometry doing the work of learning. The shape of the space itself evolves to accommodate the structure of the world.
:::

Following fission, the metric tensor $G$ must adapt to the new chart structure. We introduce a geometric flow that relaxes the metric toward consistency with the expanded ontology.

:::{prf:definition} Ontological Ricci Flow
:label: def-ontological-ricci-flow

Let $G_{ij}(z, s)$ be the capacity-constrained metric (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) parameterized by flow time $s$. Define the **local stress field** $\Xi(z) := \mathbb{E}[\Xi \mid K = k(z)]$, where $k(z)$ is the chart containing $z$. The **Ontological Ricci Flow** is:

$$
\frac{\partial G_{ij}}{\partial s} = -2\left(R_{ij} - \frac{1}{2}R\, G_{ij} + \Lambda G_{ij} - \kappa T_{ij}\right) + \nu \nabla_i \nabla_j \Xi(z),

$$
where:
- $R_{ij}$ is the Ricci curvature tensor, $R = G^{ij}R_{ij}$ the scalar curvature
- $\Lambda, \kappa$ are constants from Theorem {prf:ref}`thm-capacity-constrained-metric-law`
- $T_{ij}$ is the risk tensor
- $\nu > 0$ is the stress-curvature coupling constant

*Units:* $[\partial G / \partial s] = [z]^{-2}$; $[\Xi] = \text{nat}$; $[\nabla_i \nabla_j \Xi] = \text{nat}/[z]^2$.

*Interpretation.* The first term drives the metric toward the capacity-constrained fixed point. The second term $\nu \nabla_i \nabla_j \Xi$ introduces curvature in regions of high stress gradient, expanding the metric where new distinctions are needed.

:::

(pi-ricci-flow)=
::::{admonition} Physics Isomorphism: Ricci Flow
:class: note

**In Physics:** Hamilton's Ricci flow evolves a Riemannian metric toward constant curvature: $\partial_t g_{ij} = -2R_{ij}$. It was used by Perelman to prove the Poincare conjecture {cite}`hamilton1982ricci,perelman2002entropy`.

**In Implementation:** The Ontological Ricci Flow (Definition {prf:ref}`def-ontological-ricci-flow`) evolves the latent metric:

$$
\frac{\partial G_{ij}}{\partial s} = -2\left(R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} - \kappa T_{ij}\right) + \nu\nabla_i\nabla_j\Xi

$$
**Correspondence Table:**

| Differential Geometry | Agent (Ontological Flow) |
|:----------------------|:-------------------------|
| Metric evolution $\partial_t g$ | Metric adaptation $\partial_s G$ |
| Ricci curvature $R_{ij}$ | Ricci curvature of $G$ |
| Flow singularities | Chart fission events |
| Entropy monotonicity | Ontological stress reduction |

**Fixed Point:** The capacity-constrained metric law + vanishing stress Hessian.
::::

:::{prf:proposition} Fixed Points of Ontological Ricci Flow
:label: prop-fixed-points-of-ontological-ricci-flow

The flow has fixed points when:
1. The capacity-constrained metric law is satisfied: $R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa T_{ij}$
2. The Ontological Stress has vanishing Hessian: $\nabla_i \nabla_j \Xi = 0$

Condition (2) is satisfied when either $\Xi$ is constant (uniform stress) or $\Xi = 0$ (no stress).

*Computational Proxy.* In practice, we do not solve the Ricci flow PDE. The squared residual of the fixed-point condition can be used as a regularization loss:

$$
\mathcal{L}_{\text{Ricci}} := \left\|R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} - \kappa T_{ij}\right\|_F^2 + \nu^2 \|\nabla_i \nabla_j \Xi\|_F^2,

$$
encouraging the learned metric to satisfy the capacity constraint while penalizing stress gradients.

:::



(sec-diagnostic-nodes-a)=
## Diagnostic Nodes 49--50

:::{div} feynman-prose
Now let's get practical. How do we actually monitor whether the agent needs to expand its ontology?

We introduce two diagnostic nodes---essentially, health monitors for the conceptual structure. Node 49 watches for ontological stress (is there predictable structure hiding in the noise?). Node 50 watches whether fission is warranted (is the stress high enough, and is the expected benefit positive?).

These are the sensory neurons of the meta-learning system. They tell the agent when its current concepts are struggling and when it's time to grow.
:::

Following the diagnostic node convention ({ref}`sec-theory-thin-interfaces`), we define two new monitors for ontological expansion.

(node-49)=
**Node 49: OntologicalStressCheck**

| **#**  | **Name**                   | **Component** | **Type**                     | **Interpretation**        | **Proxy**                                                               | **Cost**                      |
|--------|----------------------------|---------------|------------------------------|---------------------------|-------------------------------------------------------------------------|-------------------------------|
| **49** | **OntologicalStressCheck** | Atlas         | Representational Sufficiency | Is texture unpredictable? | $\Xi := I(z_{\text{tex},t}; z_{\text{tex},t+1} \mid K_t, z_{n,t}, K^{\text{act}}_t)$ | $O(B \cdot d_{\text{tex}}^2)$ |

**Interpretation:** Monitors the conditional mutual information between consecutive texture components. High $\Xi$ indicates the texture channel contains predictable structure that should be in the macro-state.

**Threshold:** $\Xi < \Xi_{\text{tol}}$ (typical default $\Xi_{\text{tol}} = 0.1$ nat).

**Trigger conditions:**
- High OntologicalStressCheck ($\Xi > \Xi_{\text{tol}}$): The current chart structure is insufficient. Consider query fission (Definition {prf:ref}`def-query-fission`).
- Persistent high stress after fission: The fission direction $u$ may be suboptimal; recompute via PCA on high-stress keys.

**Computational Proxy:** Estimate $\Xi$ via a variational bound using a learned texture predictor:

$$
\hat{\Xi} = \mathbb{E}\left[\log p_\phi(z_{\text{tex},t+1} \mid z_{\text{tex},t}, K_t, z_{n,t}, K^{\text{act}}_t) - \log p_\phi(z_{\text{tex},t+1} \mid K_t, z_{n,t}, K^{\text{act}}_t)\right],

$$
where $p_\phi$ is a small MLP. If $\hat{\Xi} \approx 0$, texture is unpredictable and the firewall holds.

*Cross-reference:* Extends TextureFirewallCheck (Node 29) from measuring $\|\partial_{z_{\text{tex}}} \dot{z}\|$ (static leak) to measuring $I(z_{\text{tex},t}; z_{\text{tex},t+1})$ (temporal structure).



(node-50)=
**Node 50: FissionReadinessCheck**

| **#**  | **Name**                  | **Component** | **Type**            | **Interpretation**      | **Proxy**                                                                                                        | **Cost**         |
|--------|---------------------------|---------------|---------------------|-------------------------|------------------------------------------------------------------------------------------------------------------|------------------|
| **50** | **FissionReadinessCheck** | Atlas         | Expansion Criterion | Should ontology expand? | $\mathbb{I}(\Xi > \Xi_{\text{crit}}) \cdot \mathbb{I}(\Delta V_{\text{proj}} > \mathcal{C}_{\text{complexity}})$ | $O(N_c \cdot B)$ |

**Interpretation:** Monitors both conditions of the fission criterion (Theorem {prf:ref}`thm-fission-criterion`). Returns 1 if fission is warranted, 0 otherwise.

**Threshold:** Binary---if FissionReadinessCheck = 1, initiate query fission.

**Trigger conditions:**
- FissionReadinessCheck = 1: Execute query fission procedure.
- FissionReadinessCheck = 0 but $\Xi$ increasing: Pre-emptively compute fission direction for warm-start.

**Remediation:**
- If repeatedly triggering fission: The base architecture may be too constrained. Increase $N_v$ (codes per chart) before increasing $N_c$ (chart count).
- If fission fails to reduce $\Xi$: The fission direction missed the relevant structure. Use supervised signal (if available) to guide $u$.



(sec-summary-the-lifecycle-of-an-ontology)=
## Summary: The Lifecycle of an Ontology

:::{div} feynman-prose
Let me step back and describe the complete lifecycle of concepts in this system. It's like the metabolism of ideas---concepts are born, they live, and (as we'll see in the next section) they can also die.

The story starts in equilibrium. The agent has learned to separate signal from noise. Its discrete categories $K$ capture the task-relevant structure. The texture residual $z_{\text{tex}}$ is white noise---unpredictable, as it should be. Ontological stress $\Xi$ is near zero. All is well.

Then the world changes. New observations arrive that don't fit the existing categories. The encoder does its best, but the residuals start showing structure. Maybe a new type of entity has appeared. Maybe an old entity is behaving in new ways. Whatever the cause, texture becomes predictable. Stress rises.

As stress accumulates, observations that can't be classified start piling up at the vacuum---the "I don't know" state at the center of the disk. This is a pressure point: lots of information compressed into a single location.

Eventually, the stress exceeds the critical threshold. The vacuum becomes unstable. Like a cell dividing, the single unclassified category splits into two. A new query vector is born, pointing in the direction of maximum discriminability. The daughters separate, each claiming its share of the contested territory.

Finally, the metric relaxes. The geometry of the latent space adapts to the new chart structure, giving each concept its appropriate territory. Stress decreases. The system settles into a new equilibrium, ready for the next challenge.

This is the heartbeat of learning: equilibrium, stress, bifurcation, relaxation. Over and over, the ontology grows to match the complexity of the world.
:::

**Table 30.7.1 (Ontological Expansion Summary).**

| Concept                   | Definition/Reference                                                                                              | Units      | Diagnostic |
|:--------------------------|:------------------------------------------------------------------------------------------------------------------|:-----------|:-----------|
| **Semantic Vacuum**       | $\emptyset = \{z : \lVert z\rVert = 0\}$ (Def {prf:ref}`def-semantic-vacuum`)                                     | ---          | ---          |
| **Ontological Stress**    | $\Xi = I(z_{\text{tex},t}; z_{\text{tex},t+1} \mid K_t, z_{n,t}, K^{\text{act}}_t)$ (Def {prf:ref}`def-ontological-stress`)    | nat        | Node 49    |
| **Fission Criterion**     | $\Xi > \Xi_{\text{crit}}$ AND $\Delta V > \mathcal{C}_{\text{complexity}}$ (Thm {prf:ref}`thm-fission-criterion`) | ---          | Node 50    |
| **Query Fission**         | $q_i \mapsto \{q_i + \epsilon u, q_i - \epsilon u\}$ (Def {prf:ref}`def-query-fission`)                           | ---          | ---          |
| **Bifurcation Parameter** | $\mu = \Xi - \Xi_{\text{crit}}$ (Thm {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`)               | nat        | ---          |
| **Ricci Flow**            | $\partial_s G = -2(\text{Einstein tensor}) + \nu \nabla^2 \Xi$ (Def {prf:ref}`def-ontological-ricci-flow`)        | $[z]^{-2}$ | ---          |

**The Ontological Lifecycle:**

1. **Equilibrium:** The agent separates signal ($K, z_n$) from residual ($z_{\text{tex}}$). $\Xi \approx 0$.
2. **Stress Accumulation:** New data types appear; $z_{\text{tex}}$ becomes predictable. $\Xi$ rises.
3. **Saturation:** Unclassified observations accumulate at $z=0$.
4. **Bifurcation:** Fission criterion met; new query $q_*$ instantiated.
5. **Separation:** Daughter queries separate toward equilibrium $r^*$.
6. **Stabilization:** Metric relaxes to accommodate new chart structure.

**Conclusion.** Ontological expansion is a geometric response to representational insufficiency. The framework provides a principled criterion for when to expand chart structure (Theorem {prf:ref}`thm-fission-criterion`) and predicts the dynamics of chart separation via pitchfork bifurcation (Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`).

:::{admonition} Connection to RL #13: EWC as Degenerate Atlas
:class: note
:name: conn-rl-13
**The General Law (Fragile Agent):**
The agent maintains an **Atlas of Charts** $\{(\mathcal{U}_i, \phi_i)\}_{i=1}^{N_c}$:
- New tasks trigger **Fission**: create new chart $\mathcal{U}_{N_c+1}$ via pitchfork bifurcation
- Old charts are **topologically isolated**: transition maps prevent gradient flow between charts
- Parameters in different charts don't interfere

**The Degenerate Limit:**
Force the agent to use a **single chart** (one neural network). Add a quadratic penalty to prevent weights from moving: $\mathcal{L}_{\text{EWC}} = \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$.

**The Special Case (Continual Learning):**

$$
\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{task}} + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2

$$
This recovers **Elastic Weight Consolidation (EWC)** -- the Fisher information $F_i$ acts as an "importance weight" preventing catastrophic forgetting.

**Result:** EWC tries to cram a complex manifold into a single flat coordinate system. Catastrophic forgetting occurs because there's no **topological glue** isolating old from new -- just a soft penalty that eventually breaks under task pressure.

**What the generalization offers:**
- True isolation: chart transitions prevent gradient interference, not just penalties
- Principled expansion: new charts created when fission criterion met
- No forgetting: old charts are frozen, not elastically constrained
:::



(sec-ontological-fusion-concept-consolidation)=
## Ontological Fusion: Concept Consolidation

:::{div} feynman-prose
So far we've talked about concepts being born. But concepts can also die.

This is just as important. Without a mechanism for *removing* concepts, the system would grow without bound. Every minor variation would spawn its own category, until you had millions of tiny, overfit concepts with no generalization power.

This is the "expert explosion" problem that plagues Mixture of Experts models. They create specialists for every niche, but the specialists can't generalize because they're too specialized.

The solution is **fusion**: merging concepts that have become redundant. If two categories make the same predictions, assign the same values, and handle the same observations, why keep them separate? Merge them. Reclaim the capacity. Use it somewhere that actually needs it.

Fusion is the dual of fission. Fission creates distinctions when the data demands them. Fusion removes distinctions when they stop mattering. Together, they form a complete metabolism: concepts are born, live, and die according to their utility.
:::

*Abstract.* If Fission ({ref}`sec-symmetry-breaking-and-chart-birth`) is the birth of a concept driven by
ontological stress, **Fusion** is the death or merging of concepts driven by **metabolic efficiency**. Without Fusion,
the agent suffers from **topological heat death**: unbounded chart fragmentation where every observation eventually gets
its own private chart, destroying generalization. Fusion is triggered when the **Discrimination Gain** of keeping two
charts separate falls below the **Metabolic Cost** of maintaining them.

(rb-pruning-efficiency)=
:::{admonition} Researcher Bridge: Pruning via Metabolic Efficiency
:class: important
Most MoE (Mixture of Experts) or multi-chart models suffer from "Expert Explosion," where they create a new index for every minor variation. **Ontological Fusion** provides a principled way to forget. It merges latent charts when the **Discrimination Gain** (the information provided by keeping them separate) falls below the **Metabolic Cost** of maintaining them. It is the geometric derivation of Occam's Razor.
:::

*Cross-references:*
- Addresses Open Problem 1 from {ref}`sec-summary-the-lifecycle-of-an-ontology`.
- Dual to {ref}`sec-symmetry-breaking-and-chart-birth` (Fission).
- Connects to the Universal Governor's metabolic monitoring
  ({ref}`sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller`) and the complexity
  cost functional ({ref}`sec-the-fission-criterion`).



(sec-ontological-redundancy)=
### Ontological Redundancy

:::{div} feynman-prose
When should two concepts be merged? The answer is: when they're *functionally redundant*. But what does that mean precisely?

Two concepts are redundant when they do the same job. They occupy similar regions of belief space. They make similar predictions about the future. They assign similar values to states. In short: distinguishing between them doesn't help the agent make better decisions.

We formalize this with a redundancy measure $\Upsilon_{ij}$ that combines three components: how similar are their belief distributions? How similar are their predictions? How similar are their value assignments? If all three are similar, the concepts are redundant.

Note the exponential form: $\Upsilon_{ij}$ is 1 when concepts are identical and decays toward 0 as they differ. This gives us a smooth measure of "how redundant" rather than a binary yes/no.
:::

We define a measure of functional similarity between charts that captures whether two charts are semantically interchangeable.

:::{prf:definition} Ontological Redundancy
:label: def-ontological-redundancy

Let $K_i$ and $K_j$ be two charts with associated belief distributions $\mu_i, \mu_j$, transition models $\bar{P}_i, \bar{P}_j$, and value functions $V_i, V_j$. Their **ontological redundancy** is:

$$
\Upsilon_{ij} := \exp\left(-\left[ d_{\text{WFR}}(\mu_i, \mu_j) + D_{\mathrm{KL}}(\bar{P}_i \| \bar{P}_j) + \|V_i - V_j\|_G^2 \right]\right)

$$
where:
- $d_{\text{WFR}}(\mu_i, \mu_j)$ is the Wasserstein-Fisher-Rao distance ({prf:ref}`def-the-wfr-action`) between belief distributions,
- $D_{\mathrm{KL}}(\bar{P}_i \| \bar{P}_j) := \mathbb{E}_{k \sim \mu_i}\left[ D_{\mathrm{KL}}(\bar{P}(\cdot|k, a) \| \bar{P}_j(\cdot|k, a)) \right]$ is the mean predictive divergence,
- $\|V_i - V_j\|_G^2 := \mathbb{E}_{z \sim \mu_i}\left[ (V_i(z) - V_j(z))^2 \right]$ is the mean squared value divergence.

*Units:* Dimensionless; $\Upsilon_{ij} \in [0, 1]$.

*Interpretation:* $\Upsilon_{ij} \to 1$ implies the charts are functionally redundant: they occupy similar regions of belief space, predict similar futures, and assign similar values. $\Upsilon_{ij} \to 0$ implies they are functionally distinct.
:::



(sec-discrimination-gain)=
### Discrimination Gain

:::{div} feynman-prose
Before merging two concepts, we need to know what we'd lose. How much information about the observation stream is carried by the distinction between chart $i$ and chart $j$?

This is the **discrimination gain**: the mutual information between observations and the chart-pair distinction. If knowing whether an observation was routed to $i$ or $j$ tells you a lot about the observation, then the distinction is valuable---don't merge. If knowing the routing tells you almost nothing, the distinction is useless---merge away.

This is the MDL perspective: distinctions cost bits to encode. Keep them only if they pay for themselves in reduced distortion.
:::

Before destroying a chart, the agent must estimate the information loss.

:::{prf:definition} Discrimination Gain
:label: def-discrimination-gain

The **Discrimination Gain** $G_\Delta(i, j)$ is the mutual information the agent loses about observations by merging charts $i$ and $j$:

$$
G_\Delta(i, j) := I(X; \{K_i, K_j\}) - I(X; K_{i \cup j})

$$
where $K_{i \cup j}$ is the merged chart that routes observations previously assigned to $K_i$ or $K_j$ to a single index.

*Units:* nat.

*MDL interpretation:* $G_\Delta$ is the increase in **distortion** (description length) resulting from the merge. If $G_\Delta \approx 0$, the distinction between $K_i$ and $K_j$ carries negligible information about the observation stream.
:::

:::{prf:lemma} Redundancy-Gain Relationship
:label: lem-redundancy-gain

Under the assumption that charts partition the observation space and the encoder is deterministic given observation $x$:

$$
G_\Delta(i, j) \leq H(K_i, K_j) - H(K_{i \cup j}) = \log 2 - H(K_i | K_j) \cdot \mathbb{I}[\Upsilon_{ij} < 1]

$$
When $\Upsilon_{ij} \to 1$, the bound tightens: $G_\Delta \to 0$.

*Proof sketch.* The discrimination gain is upper-bounded by the entropy reduction from merging. When charts are redundant ($\Upsilon_{ij} \to 1$), they route to the same observations with high probability, so the conditional entropy $H(K_i | K_j) \to 0$. $\square$
:::



(sec-the-fusion-criterion)=
## The Fusion Criterion

:::{div} feynman-prose
Fusion is the mirror image of fission. Where fission is triggered by high stress and positive expected value, fusion is triggered by high redundancy and low discrimination gain.

The logic is economic. Every chart costs something to maintain: parameters, computation, increased routing complexity. If the benefit of keeping two charts separate (the discrimination gain) falls below this cost, merge them. Reclaim the resources. Use them where they matter.

But there's a subtlety: hysteresis. We don't want the system to fission, then immediately fuse, then fission again. That would be catastrophic---constant restructuring with no stable learning. The hysteresis term $\epsilon_{\text{hysteresis}}$ creates a "dead zone" where neither fission nor fusion triggers. Once you've committed to a structure, stick with it for a while.
:::

Fusion is the dual of Fission. Where Fission is triggered by high ontological stress ({prf:ref}`thm-fission-criterion`), Fusion is triggered by high redundancy and low discrimination gain.

:::{prf:axiom} Ontological Simplification Principle
:label: ax-ontological-simplification

The agent shall reduce ontological complexity when the expected value of maintaining a distinction is negative:

$$
\mathcal{C}_{\text{saved}}(N_c \to N_c - 1) > G_\Delta(i, j) + \mathbb{E}[\Delta V \mid \text{no fusion}]

$$
where $\mathcal{C}_{\text{saved}}$ is the metabolic savings from eliminating a chart.

*Remark.* This is the dual of {prf:ref}`ax-ontological-expansion-principle` (Ontological Expansion Principle). Both derive from the same MDL objective: minimize description length plus expected regret.
:::

:::{prf:theorem} Fusion Criterion
:label: thm-fusion-criterion

Charts $i$ and $j$ shall be merged if and only if:

$$
G_\Delta(i, j) < \mathcal{C}_{\text{complexity}}(N_c) - \mathcal{C}_{\text{complexity}}(N_c - 1) + \epsilon_{\text{hysteresis}}

$$
where:
- $\mathcal{C}_{\text{complexity}}(N_c) = \log N_c + \lambda_{\text{param}} |\theta_{\text{chart}}|$ is the metabolic cost of maintaining $N_c$ charts ({ref}`sec-the-fission-criterion`),
- $\epsilon_{\text{hysteresis}} > 0$ is a hysteresis constant preventing oscillatory fission-fusion ("ontological churn").

*Proof sketch.* By {prf:ref}`ax-ontological-simplification`, fusion is justified when saved complexity exceeds lost discrimination. The complexity difference is:

$$
\mathcal{C}_{\text{complexity}}(N_c) - \mathcal{C}_{\text{complexity}}(N_c - 1) = \log\frac{N_c}{N_c - 1} + \lambda_{\text{param}} |\theta_{\text{chart}}|

$$
The hysteresis term $\epsilon_{\text{hysteresis}}$ breaks the symmetry with Fission, ensuring that a chart is not immediately re-created after being destroyed. $\square$

*Remark (Units):* All terms are in nats. The criterion is dimensionally consistent.
:::



(sec-topological-collapse-the-mechanism-of-fusion)=
## Topological Collapse: The Mechanism of Fusion

:::{div} feynman-prose
Once we've decided to merge two concepts, how do we actually do it?

This isn't as simple as just deleting one. Each concept has a position in query space, a codebook of discrete symbols, and a population of observations currently assigned to it. All of these need to be carefully reconciled.

The key insight is that fusion is a *reverse bifurcation*. Where fission separated one query into two, fusion brings two queries back together. The dynamics are the same pitchfork equation, but with the sign flipped: instead of the origin being repulsive (pushing daughters apart), it becomes attractive (pulling them together).

When redundancy exceeds the critical threshold, the two query vectors start falling toward each other. Their separation shrinks until they merge into a single query at their (weighted) barycenter. The distinction between them collapses.

The observations that were split between the two charts now all go to the merged chart. The codebooks are reconciled using the jump operators we developed elsewhere. The topology simplifies. The system has forgotten a distinction it no longer needs.
:::

Once the Fusion Criterion is met, the agent must physically merge the charts. This is not simple deletion---it is **topological surgery**.



(sec-query-coalescence)=
### Query Coalescence

:::{prf:definition} Query Coalescence
:label: def-query-coalescence

Given charts $i, j$ satisfying the Fusion Criterion ({prf:ref}`thm-fusion-criterion`), the merged query is the **usage-weighted barycenter**:

$$
q_{\text{merged}} := \frac{\bar{w}_i q_i + \bar{w}_j q_j}{\bar{w}_i + \bar{w}_j}

$$
where $\bar{w}_k := \mathbb{E}[w_k(x)]$ is the historical routing weight from the Attentive Atlas ({prf:ref}`def-attentive-routing-law`).

*Interpretation:* The more frequently used chart contributes more to the merged query position. This preserves the routing behavior for the majority of observations.
:::



(sec-fiber-reconciliation-via-jump-operators)=
### Fiber Reconciliation via Jump Operators

:::{div} feynman-prose
When we merge chart $j$ into chart $i$, what happens to all the observations that were using chart $j$'s coordinate system? They need to be re-expressed in chart $i$'s coordinates.

This is where the jump operators come in. These are the "transition maps" between charts---the mathematical machinery that translates coordinates from one chart to another. We've already developed these for handling chart transitions during inference. Now we use them for ontological surgery: translating an entire population from one chart to another.

The math is the same either way: factor through a global representation, then decode into the target chart's local coordinates.
:::

When merging chart $j$ into chart $i$, observations previously routed to $j$ must be re-embedded in $i$'s coordinate system.

:::{prf:definition} Fiber Reconciliation
:label: def-fiber-reconciliation

Let $L_{j \to i}: \mathcal{F}_j \to \mathcal{F}_i$ be the factorized jump operator ({prf:ref}`def-factorized-jump-operator`). For an observation $x$ previously assigned to chart $j$ with nuisance coordinates $z_n^{(j)}$, the reconciled coordinates in chart $i$ are:

$$
z_n^{(i, \text{reconciled})} := L_{j \to i}(z_n^{(j)}) = A_i(B_j z_n^{(j)} + c_j) + d_i

$$
where $B_j$ is the chart-to-global encoder and $A_i$ is the global-to-chart decoder.

*Codebook reconciliation:* The codebook entries of chart $j$ are projected into chart $i$'s Voronoi structure. Entries that fall within existing Voronoi cells of chart $i$ are absorbed; entries that create new structure may be retained if codebook capacity permits.
:::



(sec-subcritical-bifurcation-dynamics)=
### Subcritical Bifurcation Dynamics

:::{prf:theorem} Subcritical Pitchfork for Fusion
:label: thm-subcritical-pitchfork-fusion

Let $r(s) := \|q_i(s) - q_j(s)\|$ be the query separation at computation time $s$. During fusion, the dynamics become:

$$
\frac{dr}{ds} = -(\Upsilon_{ij} - \Upsilon_{\text{crit}}) r - \alpha r^3 + \sigma\xi(s)

$$
where:
- $\Upsilon_{\text{crit}} \in (0, 1)$ is the critical redundancy threshold,
- $\alpha > 0$ is the cubic stabilization coefficient,
- $\sigma\xi(s)$ is white noise with intensity $\sigma$.

When $\Upsilon_{ij} > \Upsilon_{\text{crit}}$:
1. The linear term is **negative** (attractive toward $r = 0$).
2. $r = 0$ becomes the **unique stable attractor**.
3. The queries "fall into each other" until they merge.

*Contrast with Fission ({prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`):*

| Property                | Fission (Supercritical)          | Fusion (Subcritical)                |
|:------------------------|:---------------------------------|:------------------------------------|
| Linear term sign        | $+\mu r$ (repulsive from origin) | $-\mu r$ (attractive to origin)     |
| Trigger                 | $\Xi > \Xi_{\text{crit}}$        | $\Upsilon > \Upsilon_{\text{crit}}$ |
| Stable fixed points     | $r^* = \pm\sqrt{\mu/\alpha}$     | $r^* = 0$                           |
| Physical interpretation | Charts repel and separate        | Charts attract and merge            |

*Proof sketch.* The bifurcation structure follows from standard dynamical systems theory {cite}`strogatz2018nonlinear`. The key insight is that Fission and Fusion are **dual bifurcations**: Fission breaks $\mathbb{Z}_2$ symmetry (one chart to two); Fusion restores it (two charts to one). The sign flip in the linear term corresponds to the duality between expansion ($\Xi$) and contraction ($\Upsilon$) forces. $\square$
:::

:::{admonition} The Duality of Birth and Death
:class: feynman-added note

Notice the beautiful symmetry. Fission and fusion are governed by the *same* equation, just with opposite signs:

**Fission:** $\dot{r} = +\mu r - \alpha r^3$ (repel from origin)

**Fusion:** $\dot{r} = -\mu r - \alpha r^3$ (attract to origin)

In fission, the origin is unstable---concepts want to separate. In fusion, the origin is stable---concepts want to merge. The control parameter is different ($\Xi$ for fission, $\Upsilon$ for fusion), but the geometry is the same.

This is the topological heartbeat: expand, contract, expand, contract. The system breathes, growing complexity when the world demands it, shedding complexity when it becomes redundant.
:::



(sec-diagnostic-nodes-fusion-and-codebook-liveness)=
## Diagnostic Nodes 54--55: Fusion and Codebook Liveness

We introduce two new diagnostic nodes for the Sieve ({ref}`sec-diagnostics-stability-checks`).



:::{prf:definition} Node 54 --- FusionReadinessCheck
:label: node-fusion-readiness-check

**Component:** Atlas (Chart Router)

**Type:** Metabolic Efficiency

**Interpretation:** Are any two charts functionally redundant?

**Proxy:**

$$
\text{FusionReady} := \mathbb{I}\left[ \max_{i \neq j} \Upsilon_{ij} > \Upsilon_{\text{crit}} \right]

$$
**Computational cost:** $O(N_c^2)$ pairwise comparisons.

**Trigger condition:** Two or more charts have redundancy exceeding threshold.

**Remediation:**
1. Identify most redundant pair $(i^*, j^*) = \arg\max_{i \neq j} \Upsilon_{ij}$.
2. Verify Fusion Criterion ({prf:ref}`thm-fusion-criterion`).
3. If satisfied, initiate subcritical bifurcation dynamics.
4. Execute Query Coalescence and Fiber Reconciliation.
5. Decrement chart count: $N_c \to N_c - 1$.
:::



:::{prf:definition} Node 55 --- CodebookLivenessCheck
:label: node-codebook-liveness-check

**Component:** Codebook (VQ Layer)

**Type:** Dead Code Detection

**Interpretation:** Are any code indices unused?

**Proxy:**

$$
\text{DeadCodeDetected} := \mathbb{I}\left[ \min_k P(K = k) < \epsilon_{\text{dead}} \right]

$$
where $P(K = k)$ is the empirical usage frequency of code $k$ over a trailing window.

**Computational cost:** $O(|\mathcal{K}|)$.

**Trigger condition:** Code usage falls below minimum threshold (default $\epsilon_{\text{dead}} = 10^{-4}$).

**Remediation:** Execute Lazarus Protocol ({prf:ref}`alg-lazarus`).

*Connection to existing diagnostics:* This node operationalizes the dead-code tolerance constraint from {ref}`sec-calibrating-tolerances`: $H(K) \geq \log((1 - \rho_{\text{dead}})|\mathcal{K}|)$.
:::

**Summary Table:**

| #      | Name                  | Component | Type                 | Proxy                                                    | Cost       |
|--------|-----------------------|-----------|----------------------|----------------------------------------------------------|------------|
| **54** | FusionReadinessCheck  | Atlas     | Metabolic Efficiency | $\max_{i \neq j} \Upsilon_{ij} > \Upsilon_{\text{crit}}$ | $O(N_c^2)$ |
| **55** | CodebookLivenessCheck | Codebook  | Dead Code Detection  | $\min_k P(K=k) < \epsilon_{\text{dead}}$                 | $O(\lvert\mathcal{K}\rvert)$ |



(sec-symbolic-metabolism-intra-chart-fission-and-fusion)=
## Symbolic Metabolism: Intra-Chart Fission and Fusion

:::{div} feynman-prose
So far we've talked about the lifecycle of *charts*---the macro-categories that partition observation space. But there's another level of structure: the *symbols* within each chart.

Each chart has a codebook of discrete symbols---think of them as sub-categories within a category. "Dog" might be one chart, but within it you have symbols for different breeds, poses, lighting conditions. These symbols also have a metabolism. They can split (when a single symbol is overloaded with distinct sub-populations) and merge (when two symbols become functionally indistinguishable).

This creates a two-level hierarchy: chart metabolism at the macro level, symbol metabolism at the meso level. The same principles apply at both levels---stress triggers fission, redundancy triggers fusion---but the geometry is different. Charts live on the hyperbolic disk; symbols live in Euclidean Voronoi cells.

Understanding both levels is crucial for building systems that maintain the right level of granularity. Too few symbols and you're underfit. Too many and you're overfit. The metabolism keeps the balance.
:::

While Sections 30.1--30.11 address **chart-level** (macro) topology, the codebook symbols **within** each chart also require lifecycle management. This creates a two-level metabolic hierarchy.



(sec-symbol-fission-cluster-splitting)=
### Symbol Fission: Cluster Splitting

Symbol fission occurs when a single code index $k$ is **overloaded**---representing two or more geometrically distinct clusters.

:::{prf:definition} Intra-Symbol Variance (Geometric Tension)
:label: def-intra-symbol-variance

For code $e_k$ in chart $i$, the **geometric tension** is:

$$
\sigma_k^2 := \mathbb{E}\left[ \|z_e - e_k\|^2 \;\Big|\; \text{VQ}(z_e) = k \right]

$$
where $z_e$ is the pre-quantized encoder output.

*Units:* $[z]^2$ (squared latent units).

*Interpretation:* High $\sigma_k^2$ indicates the symbol is overloaded---its Voronoi cell contains multiple distinct clusters that should be separated.
:::

:::{div} feynman-prose
The geometric tension $\sigma_k^2$ is just the average squared distance from embedded points to their assigned codebook entry. If this is high, it means the symbol is trying to represent things that are spread out---a sign that it should split.

The procedure is simple: find the direction of maximum spread (principal eigenvector), and split the symbol along that direction. One daughter gets the points on one side; the other gets the points on the other side.
:::

**Symbol Fission Mechanism:**

1. **Detect tension:** If $\sigma_k^2 > \sigma_{\text{crit}}^2$, mark code $k$ for fission.
2. **Compute split direction:** Find the principal eigenvector $v_1$ of the conditional covariance:

   $$
   \Sigma_k := \mathbb{E}\left[ (z_e - e_k)(z_e - e_k)^\top \;\Big|\; \text{VQ}(z_e) = k \right]

   $$
3. **Instantiate daughter codes:**

   $$
   e_{k,+} := e_k + \epsilon v_1, \qquad e_{k,-} := e_k - \epsilon v_1

   $$
   where $\epsilon = \sqrt{\lambda_1 / 2}$ and $\lambda_1$ is the principal eigenvalue.
4. **Capacity check:** If the codebook is full, trigger Symbol Fusion elsewhere to free a slot.



(sec-symbol-fusion-synonym-merging)=
### Symbol Fusion: Synonym Merging

Symbol fusion is the **generalization** step---merging symbols that are functionally indistinguishable.

:::{prf:definition} Functional Indistinguishability
:label: def-functional-indistinguishability

Two symbols $k_1, k_2$ within the same chart are fusion candidates if the **policy divergence** and **value gap** are negligible:

$$
\mathcal{D}_f(k_1, k_2) := D_{\mathrm{KL}}\left( \pi(\cdot | k_1) \| \pi(\cdot | k_2) \right) + |V(k_1) - V(k_2)|

$$
If $\mathcal{D}_f(k_1, k_2) < \epsilon_{\text{indist}}$, the distinction provides no **control authority**.

*Units:* nat.

*Interpretation:* Symbols are functionally indistinguishable when the policy and value function treat them identically.
:::

:::{div} feynman-prose
The criterion for symbol fusion is beautifully pragmatic: if the policy doesn't distinguish between two symbols---if it takes the same actions with the same expected values---then the distinction is meaningless. Merge them.

This is the "functional" perspective. We don't care about geometric similarity in the latent space. We care about behavioral similarity in the decision space. Two symbols that look different but act the same are synonyms and should be merged.
:::

**Symbol Fusion Mechanism:**

1. **Coalesce embeddings:**

   $$
   e_{\text{merged}} := \frac{1}{2}(e_{k_1} + e_{k_2})

   $$
2. **Remap transitions:** Update all entries in the world model $\bar{P}$ that reference $k_1$ or $k_2$ to point to the merged index.
3. **Free slot:** Return one index to the available pool for future Symbol Fission.



(sec-the-lazarus-protocol-dead-code-reallocation)=
### The Lazarus Protocol: Dead Code Reallocation

:::{div} feynman-prose
There's a pathology in vector quantization called "codebook collapse." Some codes get used constantly, while others are never used at all. The dead codes are wasted capacity---you're paying for symbols that don't represent anything.

The Lazarus Protocol resurrects dead codes. It takes a code that nobody's using and moves it to where it's needed: next to the most overloaded code, ready to take over half its population.

This is entropy redistribution. Information is piling up in some regions and absent from others. The protocol redistributes the representational capacity to match the data distribution.
:::

In standard VQ-VAEs, **codebook collapse** is a major failure mode where most codes are never used. The Lazarus Protocol recycles dead codes to high-information-density regions.

:::{prf:algorithm} Lazarus Reallocation
:label: alg-lazarus

**Input:** Dead code $k_{\text{dead}}$ with $P(K = k_{\text{dead}}) < \epsilon_{\text{dead}}$.

**Procedure:**
1. Find the most stressed symbol:

   $$
   k_{\text{stressed}} := \arg\max_k \sigma_k^2

   $$
2. Perform Symbol Fission on $k_{\text{stressed}}$, reusing index $k_{\text{dead}}$:
   - Compute split direction $v_1$ from $\Sigma_{k_{\text{stressed}}}$.
   - Set $e_{k_{\text{dead}}} := e_{k_{\text{stressed}}} + \epsilon v_1$.
   - Update $e_{k_{\text{stressed}}} := e_{k_{\text{stressed}}} - \epsilon v_1$.
3. Update Voronoi cells: The new code inherits half of $k_{\text{stressed}}$'s cell.

**Effect:** Vocabulary migrates to high-information-density regions. Dead codes are "resurrected" where they are needed.

*Connection to existing constraints:* This implements the anti-collapse regularizer from {ref}`sec-calibrating-tolerances`: $\lambda_{\text{use}} D_{\mathrm{KL}}(\hat{p}(K) \| \text{Unif}(\mathcal{K}))$.
:::



(sec-measure-theoretic-formalization)=
### Measure-Theoretic Formalization

For maximum rigor, we treat the codebook not as a static list of vectors but as a **discrete measure** on the fiber. This enables a variational characterization of code allocation.

:::{prf:definition} Symbolic Voronoi Partition
:label: def-voronoi-partition

Let $\mathcal{Z}_i$ be the continuous fiber associated with chart $i$. The codebook $\mathcal{C}_i = \{e_{i,k}\}_{k=1}^{N_v}$ induces a partition $\{\mathcal{V}_k\}$ of $\mathcal{Z}_i$ via:

$$
\mathcal{V}_k := \left\{ z \in \mathcal{Z}_i : d_G(z, e_k) \leq d_G(z, e_j) \;\forall j \neq k \right\}

$$
The probability mass of symbol $k$ is the measure of its Voronoi cell:

$$
P(k) := \int_{\mathcal{V}_k} p(z)\, d\mu_G(z)

$$
where $d\mu_G = \sqrt{\det G}\, dz$ is the Riemannian volume form.
:::

:::{prf:definition} Local Distortion Functional
:label: def-local-distortion

The **local distortion** of symbol $k$ quantifies the representational error within its Voronoi cell:

$$
\mathcal{D}_k := \int_{\mathcal{V}_k} d_G(z, e_k)^2\, p(z)\, d\mu_G(z)

$$
*Units:* $[z]^2$ (weighted squared geodesic distance).

*Relation to geometric tension:* $\mathcal{D}_k = P(k) \cdot \sigma_k^2$, where $\sigma_k^2$ is the intra-symbol variance ({prf:ref}`def-intra-symbol-variance`).
:::

:::{prf:definition} Symbol Utility Functional
:label: def-symbol-utility

The **utility** $U_k$ of symbol $k$ measures its contribution to control authority and predictive accuracy:

$$
U_k := P(k) \cdot I(K=k; A) + P(k) \cdot I(K=k; K_{t+1})

$$
where:
- $I(K=k; A)$ is the mutual information between symbol activation and action selection,
- $I(K=k; K_{t+1})$ is the mutual information between symbol activation and next-state prediction.

*Units:* nat.

*Interpretation:* A symbol with $U_k \approx 0$ neither influences actions nor aids prediction---it is **semantically dead** regardless of its usage frequency.
:::

:::{prf:theorem} Optimal Reallocation Gradient
:label: thm-reallocation-gradient

Let $k_{\text{dead}}$ satisfy $U_{k_{\text{dead}}} < \epsilon_U$ and let $k_{\text{stressed}}$ satisfy $\mathcal{D}_{k_{\text{stressed}}} = \max_k \mathcal{D}_k$. The expected reduction in global distortion per reallocated code is:

$$
\frac{\delta \mathcal{D}}{\delta N_{\text{codes}}} \approx \frac{\mathcal{D}_{k_{\text{stressed}}}}{H(K = k_{\text{stressed}})}

$$
*Proof sketch.* In the high-resolution limit of vector quantization (Zador's theorem {cite}`zador1982asymptotic`), distortion scales as $\mathcal{D} \propto N_v^{-2/d}$ where $d$ is the latent dimension. Reallocating a code from a zero-utility region to a high-distortion region maximizes the gradient of the distortion functional. The denominator $H(K = k_{\text{stressed}})$ normalizes by the information content of the target symbol. $\square$
:::
:::{prf:corollary} The Bimodal Instability Theorem (Fission Trigger)
:label: cor-bimodal-instability

Let $K$ be a macro-symbol with associated policy $\pi(\cdot|K)$. The **Structural Stability** of $K$ is inversely proportional to its Varentropy.

If the policy $\pi(\cdot|K)$ is a mixture of two disjoint, equally weighted strategies (a "Buridan's Ass" scenario on a value ridge), the Varentropy satisfies:

$$
V_H(K) = \frac{1}{4}\left(\frac{\Delta Q}{T_c}\right)^2,

$$
where $\Delta Q = |Q_1 - Q_2|$ is the value gap between the modes. In the limit of distinct modes ($\Delta Q \gg T_c$), $V_H$ is maximized, whereas for a uniform (maximum entropy) distribution, $V_H = 0$.

*Units:* $\mathrm{nat}^2$.

**Refined Fission Criterion:**
The **Geometric Tension** $\sigma_k^2$ (Definition {prf:ref}`def-intra-symbol-variance`) is rigorously generalized by the **Varentropy Excess**:

$$
\text{Fission}(K) \iff V_H(K) > \mathcal{V}_{\text{crit}} \quad \text{AND} \quad H(K) > H_{\text{noise}}.

$$
**Interpretation:**
- **High $H$, Low $V_H$:** Aleatoric Uncertainty (Noise/Fog). The distribution is flat. *Action:* Smoothing/Integration.
- **High $H$, High $V_H$:** Epistemic Conflict (Bifurcation). The distribution is multimodal. *Action:* Topological Fission (Node 50).

*Proof:* See Appendix {ref}`E.9 <sec-appendix-e-proof-of-corollary-bimodal-instability>`.

:::

:::{admonition} Varentropy: The Key Diagnostic
:class: feynman-added important

This corollary is crucial. It says that not all uncertainty is the same.

**High entropy, low varentropy:** The distribution is spread out but *smooth*. This is aleatoric uncertainty---genuine randomness in the world. You shouldn't split; there's nothing to split *on*.

**High entropy, high varentropy:** The distribution is spread out and *lumpy*. This is epistemic uncertainty---you're confused between distinct alternatives. This is exactly when you should split: the lumps become your daughter concepts.

Varentropy detects multimodality. It's the variance of the surprise values. A flat distribution has zero varentropy (all surprises are equal). A bimodal distribution has high varentropy (some outcomes are very surprising, others not at all).

This is the key to knowing *when* fission is appropriate: not just when entropy is high, but when varentropy is high.
:::



(sec-comparison-chart-vs-symbol-metabolism)=
## Comparison: Chart vs. Symbol Metabolism

The fission/fusion dynamics operate at two hierarchical levels with analogous but distinct forces.

**Table 30.13.1 (Two-Level Metabolic Hierarchy).**

| Level             | Object          | Expansion Force                | Contraction Force                    | Geometry                   | Diagnostic       |
|:------------------|:----------------|:-------------------------------|:-------------------------------------|:---------------------------|:-----------------|
| **Chart** (Macro) | Query $q_i$     | Ontological Stress $\Xi$       | Redundancy $\Upsilon_{ij}$           | Hyperbolic (Poincare disk) | Nodes 49, 50, 54 |
| **Symbol** (Meso) | Embedding $e_k$ | Geometric Tension $\sigma_k^2$ | Indistinguishability $\mathcal{D}_f$ | Euclidean (Voronoi cell)   | Node 55          |

**Key distinctions:**

1. **Chart metabolism** governs the **global manifold partition**---how many semantic categories exist.
2. **Symbol metabolism** governs the **local tessellation within each chart**---how finely each category is discretized.
3. The **Universal Governor** ({ref}`sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller`) monitors both levels via the total entropy budget:

   $$
   H(K_{\text{chart}}) + \mathbb{E}_{i}[H(K_{\text{code}} | K_{\text{chart}} = i)] \leq B_{\text{metabolic}}

   $$


(sec-summary-the-topological-heartbeat)=
## Summary: The Topological Heartbeat

:::{div} feynman-prose
Let me give you the big picture. The agent's ontology breathes.

In systole, the system *expands*. Stress accumulates. New observations don't fit existing categories. The texture channel becomes predictable. The bifurcation criterion triggers. A chart splits. Complexity increases. The manifold grows a new dimension of meaning.

In diastole, the system *contracts*. Usage patterns shift. Two categories start doing the same work. Redundancy accumulates. The fusion criterion triggers. Charts merge. Complexity decreases. The manifold sheds a distinction it no longer needs.

Expand, contract, expand, contract. The system breathes, maintaining the right level of complexity for the current task. Not too simple (underfitting), not too complex (overfitting). Just right.

This is homeostasis at the level of concepts. The Universal Governor watches the complexity budget, the discrimination floor, the liveness constraints. It keeps the ontology healthy---neither starving nor bloated.

And the beautiful thing is that this all emerges from simple principles. Minimize description length plus expected regret. Create distinctions when they pay for themselves. Remove distinctions when they don't. The rest is geometry.
:::

The complete ontological lifecycle forms a **homeostatic cycle**:

**Table 30.14.1 (The Ontological Heartbeat).**

| Phase                 | Trigger                             | Mechanism                                                                                 | Effect                                |
|:----------------------|:------------------------------------|:------------------------------------------------------------------------------------------|:--------------------------------------|
| **Systole (Fission)** | $\Xi > \Xi_{\text{crit}}$           | Supercritical bifurcation ({prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`) | $N_c \to N_c + 1$; manifold expands   |
| **Diastole (Fusion)** | $\Upsilon > \Upsilon_{\text{crit}}$ | Subcritical bifurcation ({prf:ref}`thm-subcritical-pitchfork-fusion`)                     | $N_c \to N_c - 1$; manifold contracts |

The {ref}`Universal Governor (Section 26) <sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller>` maintains homeostasis by monitoring:

1. **Complexity budget:** $H(K_{\text{chart}}) + H(K_{\text{code}}) \leq B_{\text{metabolic}}$.
2. **Discrimination floor:** $G_\Delta(i, j) > G_{\min}$ for all retained chart pairs.
3. **Liveness constraint:** $P(K = k) > \epsilon_{\text{dead}}$ for all active codes.

**Conclusion.** By adding Fusion to Fission, the agent possesses a complete **topological metabolism**. Fission creates structure when the world demands finer distinctions; Fusion destroys structure when distinctions become redundant. The balance is governed by the same MDL principle that drives the entire framework: minimize description length plus expected regret.

:::{prf:proposition} Equipartition of Meaning
:label: prop-equipartition

At metabolic equilibrium, the marginal utility per bit is uniform across the ontological hierarchy:

$$
\frac{\partial U}{\partial H(K_{\text{chart}})} \approx \frac{\partial U}{\partial H(K_{\text{code}})} \approx \text{const.}

$$
where $U$ is the total utility functional (value minus complexity cost).

*Interpretation:* The agent allocates representational capacity such that one additional bit of chart-level information provides the same marginal value as one additional bit of symbol-level information. This is the information-theoretic analogue of thermodynamic equipartition.
:::



(sec-thermodynamic-hysteresis-calibration)=
## Thermodynamic Calibration of Ontological Hysteresis

:::{div} feynman-prose
Here's a practical question: how do we set the hysteresis threshold $\epsilon_{\text{hysteresis}}$?

Too small, and the system oscillates---fission, then fusion, then fission again, wasting computation on structural churn. Too large, and the system becomes rigid---unable to adapt when it genuinely needs to.

The answer comes from thermodynamics. Fission and fusion aren't free. They cost energy. Creating a new chart requires initializing parameters (Landauer's principle: erasing random bits requires work). Destroying a chart requires erasing the distinction (also requires work). Any fission-fusion cycle that accomplishes nothing still pays these costs.

The hysteresis threshold must be at least as large as this minimum metabolic cost. Otherwise the system could spontaneously chatter---fission and fuse repeatedly---dissipating energy without accomplishing anything.

This is thermodynamic calibration: using fundamental physical principles to set practical hyperparameters.
:::

We derive the hysteresis constant $\epsilon_{\text{hysteresis}}$ appearing in the Fusion Criterion ({prf:ref}`thm-fusion-criterion`) as a thermodynamic necessity arising from the computational metabolism of the agent ({ref}`sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics`).

:::{prf:theorem} Thermodynamic Lower Bound on Hysteresis
:label: thm-thermodynamic-hysteresis-bound

Let $\mathcal{C}$ be a cycle of ontological operations consisting of a fission event $N_c \to N_c + 1$ followed immediately by a fusion event $N_c + 1 \to N_c$. Let $T_c$ be the cognitive temperature and $\mathcal{W}_{\text{comp}}$ be the metabolic work of parameter instantiation. To satisfy the generalized Second Law of Thermodynamics for open cognitive systems (Theorem {prf:ref}`thm-generalized-landauer-bound`), the hysteresis threshold must satisfy:

$$
\epsilon_{\text{hysteresis}} \geq \frac{1}{\beta_{\text{eff}}} \left( \Delta H_{\text{Shannon}} + \frac{1}{T_c}\mathcal{W}_{\text{comp}} \right)

$$
where $\beta_{\text{eff}} = 1/T_c$ is the inverse cognitive temperature and $\Delta H_{\text{Shannon}}$ is the entropy reduction associated with the discarded distinction.

*Proof.*
Consider the free energy functional $\mathcal{F} = E - T_c S$.

1. **Fission Cost:** The creation of a new chart requires initializing a set of parameters $\theta_{\text{new}}$. By Landauer's Principle ({ref}`Landauer's Principle <pi-landauer-principle>`), the erasure of the previous random state of these memory units to a low-entropy initialization requires work $\mathcal{W}_{\text{init}} \geq k T_c \ln 2 \cdot |\theta_{\text{new}}|$.

2. **Fusion Cost:** The merger of two charts implies the erasure of the mutual information $I(X; \{K_i, K_j\}) - I(X; K_{i \cup j})$, defined as the Discrimination Gain $G_\Delta$ ({prf:ref}`def-discrimination-gain`). This is an irreversible logical operation, dissipating heat $Q_{\text{fus}} \geq T_c G_\Delta$.

3. **Cycle Condition:** For the cycle $\mathcal{C}$ to be non-spontaneous (preventing chattering), the total free energy change must be positive. The Governor imposes a metabolic efficiency constraint $\eta_{\text{ROI}} > \eta_{\min}$ ({ref}`sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller`).

4. **Derivation:** The utility gain of the cycle is zero (the topology is unchanged). The cost is $\mathcal{W}_{\text{init}} + Q_{\text{fus}}$. For the cycle to be rejected by the Fusion Criterion ({prf:ref}`thm-fusion-criterion`), the hysteresis term must exceed the minimum metabolic dissipation of the cycle:

$$
\epsilon_{\text{hysteresis}} \geq \inf_{\mathcal{C}} \oint \dot{\mathcal{M}}(s) ds

$$
Substituting the Landauer bound yields the stated inequality. $\square$
:::

*Units:* $[\epsilon_{\text{hysteresis}}] = \text{nat}$, consistent with the complexity cost functional.

*Cross-references:* This resolves the hysteresis calibration question by grounding it in the Landauer thermodynamics of
{ref}`sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics`.



(sec-hyperbolic-coalescence)=
## Intrinsic Coalescence on Hyperbolic Manifolds

:::{div} feynman-prose
There's a subtlety we glossed over earlier. When we merge two charts, we compute the barycenter of their query vectors. But what does "barycenter" mean in hyperbolic space?

In Euclidean space, it's simple: the barycenter is the average. But the Poincare disk isn't Euclidean. Straight lines in Euclidean space are curved geodesics on the disk. The "average" computed in Euclidean coordinates isn't geometrically meaningful---it doesn't minimize geodesic distances.

The correct notion is the Frechet mean: the point that minimizes the sum of squared geodesic distances to all the inputs. In hyperbolic space, this is different from the Euclidean average, and we need to compute it properly using Riemannian gradient descent.

This is the kind of detail that seems pedantic until you get it wrong. If you use Euclidean averaging in hyperbolic space, the merged query ends up in the wrong place, and routing becomes systematically biased. Getting the geometry right matters.
:::

The Query Coalescence operation ({prf:ref}`def-query-coalescence`) uses a Euclidean barycenter $\bar{q} = \frac{1}{N}\sum q_i$. In the Poincare disk $\mathbb{D}$, this induces geometric distortion since straight lines in $\mathbb{R}^n$ are not geodesics in $\mathbb{D}$. The rigorous fusion operator is the **Frechet Mean**, following the pattern established in {prf:ref}`def-class-centroid-in-poincar-disk`.

:::{prf:definition} Hyperbolic Frechet Mean for Query Coalescence
:label: def-hyperbolic-frechet-coalescence

Let $\{q_i\}_{i=1}^k \subset \mathbb{D}$ be a set of chart query vectors with associated usage weights $\bar{w}_i := \mathbb{E}[w_i(x)]$ from the Attentive Atlas ({prf:ref}`def-attentive-routing-law`). The **Intrinsic Merged Query** is:

$$
q_{\text{merged}} := \operatorname*{arg\,min}_{q \in \mathbb{D}} \sum_{i=1}^k \bar{w}_i \cdot d^2_{\mathbb{D}}(q, q_i),

$$
where $d_{\mathbb{D}}(x, y) = \operatorname{arccosh}\left(1 + \frac{2\|x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}\right)$ is the hyperbolic distance.

*Units:* $[q_{\text{merged}}] = [q_i]$ (dimensionless in the unit disk).

*Cross-reference:* This definition supersedes {prf:ref}`def-query-coalescence` for hyperbolic embeddings.
:::

:::{prf:theorem} Existence and Uniqueness of Fusion Center
:label: thm-frechet-fusion-uniqueness

Since the Poincare disk $(\mathbb{D}, G)$ is a complete, simply connected Riemannian manifold with non-positive sectional curvature ($K=-1$), it is a Hadamard space (global CAT(0) space). The squared distance function $d^2_{\mathbb{D}}(\cdot, y)$ is strictly convex. Therefore, the functional $F(q) = \sum \bar{w}_i d^2_{\mathbb{D}}(q, q_i)$ admits a unique global minimizer.

*Proof.* By Cartan's theorem on Hadamard manifolds, the distance function from any point is strictly convex along geodesics. The weighted sum of strictly convex functions is strictly convex, ensuring the minimizer exists and is unique. $\square$
:::

:::{prf:remark} Computational Algorithm
:label: rem-frechet-algorithm

The minimizer can be computed via Riemannian gradient descent:

$$
q_{t+1} = \operatorname{Exp}_{q_t}\left( -\eta \sum_i \bar{w}_i \operatorname{Log}_{q_t}(q_i) \right)

$$
where:
- $\operatorname{Exp}_p: T_p\mathbb{D} \to \mathbb{D}$ is the exponential map at $p$
- $\operatorname{Log}_p: \mathbb{D} \to T_p\mathbb{D}$ is the logarithmic map (inverse of exponential)

For the Poincare disk, these have closed-form expressions via Mobius operations ({ref}`sec-bulk-boundary-independence`).

*Complexity:* $O(k \cdot d)$ per iteration, where $k$ is the number of charts being merged and $d$ is the embedding dimension.
:::

*Cross-references:* This resolves the geometric inconsistency by ensuring coalescence respects the intrinsic hyperbolic geometry.



(sec-fission-inhibition-corollary)=
## The Fission Inhibition Corollary (Hierarchical Metabolism Resolution)

:::{div} feynman-prose
Here's a worry: if one level of the hierarchy undergoes fission, does that trigger fission at other levels? Could we have a cascade where expanding one chart triggers expansion everywhere, leading to runaway complexity?

Fortunately, no. The architecture is inherently self-stabilizing. When a coarse-level chart fissions, it *absorbs* variance that would otherwise flow to finer levels. The fine levels see *less* structure after the fission, not more.

This is the benefit of hierarchical representation. Coarse structure is captured at coarse scales, leaving only residuals for fine scales. When you add resolution at a coarse scale, you're reducing the burden on fine scales.

The formal statement is the Fission Inhibition Corollary: fission at level $\ell$ *reduces* the probability of fission at level $\ell+1$. The hierarchy is a damper, not an amplifier. Perturbations attenuate as they propagate upward.
:::

We prove that the Stacked TopoEncoder architecture ({ref}`sec-stacked-topoencoders-deep-renormalization-group-flow`) enforces **top-down stability** via the properties of the residual variance in the Renormalization Group (RG) flow. A fission event at layer $\ell$ does not trigger cascading fission at higher layers.

:::{prf:theorem} Fission Inhibition Corollary
:label: thm-fission-inhibition

Let $\mathcal{E}^{(\ell)}$ be the encoder at scale $\ell$. A Topological Fission event at layer $\ell$ (increasing chart count $N_c^{(\ell)} \to N_c^{(\ell)}+1$) strictly reduces the probability of fission at layer $\ell+1$.

*Proof.*
1. **Residual Coupling:** The input to layer $\ell+1$ is the normalized residual of layer $\ell$: $x^{(\ell+1)} = z_{\text{tex}}^{(\ell)} / \sigma^{(\ell)}$.

2. **Approximation Theory:** Fission adds a centroid to the Voronoi partition at layer $\ell$. By standard quantization theory (Zador's theorem), increasing codebook size strictly reduces the mean squared quantization error (distortion), provided the data is not uniform.

3. **Variance Reduction:** The reconstruction error $\|z_{\text{tex}}^{(\ell)}\|^2$ decreases, implying the scale factor $\sigma^{(\ell)}$ decreases.

4. **Stress Damping:** Ontological Stress at layer $\ell+1$ is upper-bounded by the mutual information of its input. Since the input variance is reduced (relative to the pre-fission state), the extractable structure $I(x^{(\ell+1)}_t; x^{(\ell+1)}_{t+1})$ decreases.

5. **Conclusion:** Macro-scale adaptation absorbs structural variance, starving the micro-scale of the stress required to trigger bifurcation. $\square$
:::

:::{prf:corollary} Hierarchical Stability
:label: cor-hierarchical-stability

The stacked architecture is **inherently stable** against fission cascades. Ontological expansion at coarse scales (low $\ell$) pre-empts the need for expansion at fine scales (high $\ell$).

*Interpretation:* If the agent learns a new high-level concept (e.g., "mammal"), the residual variance available to learn low-level distinctions (e.g., specific breeds) is reduced. The hierarchy self-regulates, preventing runaway complexity growth.
:::

*Cross-references:* This resolves the hierarchical metabolism question by showing that the RG structure naturally dampens topological perturbations from propagating upward.
