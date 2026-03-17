(sec-supervised-topology-semantic-potentials-and-metric-segmentation)=
# Supervised Topology: Semantic Potentials and Metric Segmentation

## TLDR

- Supervision is not just labels; it is a **geometric constraint** that shapes the latent metric and partitions state
  space into semantic regions.
- Class structure appears as **potentials and jump structure**: “belong together” becomes “live near each other” in
  $\mathcal{Z}$.
- This chapter links metric learning/contrastive ideas to the broader geometry/control story: classification is a
  topological segmentation problem.
- Use the derived losses and diagnostics to detect when semantics are collapsing (merged classes) or fragmenting
  (over-segmentation).
- Outputs: a principled interpretation of supervised losses as semantic field constraints, not heuristics.

## Roadmap

1. Why classification should be read as geometry/topology.
2. Semantic potentials, metric segmentation, and trainable losses.
3. Diagnostics: what “good semantic structure” looks like in $\mathcal{Z}$.

:::{div} feynman-prose
Here's a question that sounds simple but has kept me up at night: What does it mean to classify something?

The standard story goes like this: you have a picture of a cat, you feed it to a neural network, out pops "cat." Classification done. But wait---what actually *happened* in there? What does it mean for the network to "know" it's a cat?

The usual answer is something about decision boundaries and hyperplanes separating feature vectors. And that's fine as a computational description, but it misses something profound. Classification isn't just about drawing lines between categories. It's about *organizing your understanding of the world* so that things that behave similarly end up near each other, and things that behave differently end up far apart.

In this section, we're going to take that intuition seriously. We're going to see that class labels aren't just targets for prediction---they're *geometric constraints* on the shape of the agent's internal representation. When you label a bunch of examples as "cat," you're not just providing training signal. You're telling the agent: "These things belong together. They should live in the same neighborhood of your mental map."

And here's the beautiful part: this geometric view connects classification to everything else we've been building. The same metric structure that governs how the agent moves through latent space, the same potential landscapes that guide decisions, the same chart-based organization---all of it gets woven together with supervision into a unified picture.
:::

(rb-metric-learning)=
:::{admonition} Researcher Bridge: Metric Learning for Classification
:class: info
This section recasts supervised labels as geometric constraints. It is the same idea as contrastive or metric learning, but expressed as class-conditioned potentials and separation in the latent manifold.
:::

:::{div} feynman-prose
We rigorously define the role of discrete class labels $\mathcal{Y}$ within the continuous latent geometry. Rather than treating classification as "predicting a target variable," we define classification via the **Manifold Hypothesis** {cite}`carlsson2009tda`: Class labels identify topologically coherent regions of the latent manifold, and classification is **equilibrium chart assignment under class-conditioned gradient flow**.

This section **extends** the context-conditioned framework of {ref}`sec-relationship-to-the-context-conditioned-framework`, providing the topological constraints that make classification geometrically meaningful. The approach integrates ideas from topological data analysis {cite}`carlsson2009tda`, mixture-of-experts routing {cite}`shazeer2017moe`, hyperbolic embeddings {cite}`nickel2017poincare`, and Riemannian optimization {cite}`bonnabel2013rsgd`.
:::

(sec-relationship-to-the-context-conditioned-framework)=
## Relationship to the Context-Conditioned Framework

:::{div} feynman-prose
Before diving into the formalism, let me give you the big picture.

We've already established that classification can be viewed as selecting a context $c$ from the label space $\mathcal{Y}$---that's what Section 23.6 was about. The effective potential $\Phi_{\text{eff}} = -\log p(y|z)$ tells us how "expensive" it is to be at position $z$ if we're trying to be class $y$.

But that's just the beginning. The potential alone doesn't tell us whether our classification system is *well-organized*. We could have a valid potential landscape but still have a mess---different classes all jumbled together, no clear separation, brittle boundaries.

What we need are *topological constraints*: rules that enforce geometric sanity. These constraints say things like: "Charts shouldn't be confused about what class they represent," and "Different classes should be far apart in the metric," and "If you start in the cat region and flow downhill, you should stay in the cat region."

Think of it like city planning. The potential landscape is like the terrain---hills and valleys. But good city planning also requires that neighborhoods be coherent (you don't want houses interleaved with factories), that districts be separated by clear boundaries, and that traffic flows smoothly within regions. The topological constraints are the zoning laws.
:::

:::{prf:remark} Extension, Not Replacement
:label: rem-extension-not-replacement

{ref}`sec-relationship-to-the-context-conditioned-framework` establishes classification as selecting a context $c \in \mathcal{Y}$ (the label space), with effective potential $\Phi_{\text{eff}} = -\log p(y|z)$ (Theorem {prf:ref}`thm-universal-context-structure`). This section specifies the **topological constraints** that enforce geometric coherence of this classification:

1. Charts should be semantically pure (one class per chart, modulo transition regions)
2. Different classes should be metrically separated (long geodesics between class regions)
3. Classification should be stable under dynamics (regions of attraction)

:::

:::{div} feynman-prose
Now let's get precise. What does it mean for the atlas of charts to be "organized" with respect to class labels?
:::

:::{prf:definition} Semantic Partition
:label: def-semantic-partition

Let $\mathcal{Y} = \{1, \ldots, C\}$ be the set of class labels and $\mathcal{K}$ the macro-state register (Definition 2.2.1). A labeling $Y: \mathcal{X} \to \mathcal{Y}$ induces a **soft partition** of the chart atlas:

$$
\mathcal{A}_y := \{k \in \mathcal{K} : P(Y=y \mid K=k) > 1 - \epsilon_{\text{purity}}\},

$$
where $\epsilon_{\text{purity}} \in (0, 0.5)$ is the purity threshold.

*Interpretation:* $\mathcal{A}_y$ is the **sub-atlas** of charts predominantly associated with class $y$. A chart $k$ belongs to $\mathcal{A}_y$ if, given that a sample routes to chart $k$, the probability of class $y$ exceeds $1 - \epsilon_{\text{purity}}$.

:::

:::{div} feynman-prose
Let me make sure this is crystal clear. We have charts---local coordinate systems that tile the latent manifold. And we have class labels---cat, dog, car, whatever. The semantic partition asks: "Which charts are 'cat charts'? Which are 'dog charts'?"

A chart $k$ belongs to the cat sub-atlas if, whenever a sample ends up being handled by chart $k$, it's almost always a cat. The "almost always" is controlled by $\epsilon_{\text{purity}}$---if we set it to 0.1, then chart $k$ is a cat chart if at least 90% of the samples routed to it are cats.

Notice this is a *soft* partition. Some charts might be pure (100% one class), some might be impure (mixed classes), and some charts might not belong to *any* sub-atlas because they're just too confused. The impure charts are especially interesting---they're the transition regions where classification gets hard.
:::

:::{prf:proposition} Soft Injectivity
:label: prop-soft-injectivity

The sub-atlases need not be disjoint. Charts in $\mathcal{A}_i \cap \mathcal{A}_j$ for $i \neq j$ are **transition regions** characterized by:

1. **Low purity:** $\max_y P(Y=y \mid K=k) < 1 - \epsilon_{\text{purity}}$ for all $y$
2. **High entropy:** $H(Y \mid K=k) > H_{\text{transition}}$ (conditional entropy; see {cite}`cover1991elements`)
3. **Low information content:** These charts carry less semantic information per the information bottleneck principle {cite}`tishby2015ib`

*Remark (Geometric Interpretation).* Transition charts correspond to saddle regions of the semantic potential landscape---unstable fixed points between class regions of attraction.

**Cross-references:** {ref}`sec-relationship-to-the-context-conditioned-framework` (Context-Conditioned Policies), Definition 2.2.1 (Macro-State Register), {ref}`sec-tier-the-attentive-atlas` (Router Weights).

:::

:::{admonition} The Geography of Confusion
:class: feynman-added tip

Think of the latent space as a landscape with mountain ranges separating different kingdoms (classes). Each chart is like a town.

- **Pure charts** are towns deep inside a kingdom---if you're there, you're definitely in Cat Country.
- **Transition charts** are border towns, situated in mountain passes between kingdoms. These are the "I'm not sure if this is a cat or a small dog" regions.

The saddle point interpretation is key: a transition chart is like a pass at the top of a ridge. You're at a local maximum of elevation on the ridge, but if you step to either side, you'll roll down into one kingdom or the other. These are geometrically unstable positions---the slightest perturbation pushes you toward one class or another.

This is why hard examples in classification tend to cluster: they're all trying to fit through the same mountain passes!
:::

(sec-the-semantic-potential)=
## The Semantic Potential

:::{div} feynman-prose
We embed class labels into the dynamics via a **class-conditioned potential** that shapes the energy landscape.

Here's the key idea: we already have a potential landscape $V_{\text{base}}$ that the agent uses for navigation and decision-making. Now we're going to add a term that depends on the class label. This additional term acts like a "semantic gravity"---it pulls the system toward regions associated with a particular class.

Think of it like coloring a topographic map. The base potential gives you the elevation---the hills and valleys. The class-conditioned term paints the valleys in different colors: blue for cat, red for dog, green for car. The semantic potential for "cat" makes the blue regions into deeper valleys, encouraging the system to settle there.
:::

:::{prf:definition} Class-Conditioned Potential
:label: def-class-conditioned-potential

Given a target class $y \in \mathcal{Y}$, define the semantic potential:

$$
V_y(z, K) := -\beta_{\text{class}} \log P(Y=y \mid K) + V_{\text{base}}(z, K),

$$
where:
- $P(Y=y \mid K) = \text{softmax}(\Theta_{K,:})_y$ with learnable parameters $\Theta \in \mathbb{R}^{N_c \times C}$
- $V_{\text{base}}(z, K)$ is the unconditioned critic ({ref}`sec-the-hjb-correspondence`)
- $\beta_{\text{class}} > 0$ is the **class temperature** (inverse of semantic diffusion)
- Units: $[V_y] = \mathrm{nat}$

*Remark (Chart-to-Class Mapping).* The learnable parameter $\Theta_{k,y}$ represents the log-affinity of chart $k$ for class $y$. After training, $P(Y=y \mid K=k) = \text{softmax}(\Theta_{k,:})_y$ approximates the empirical conditional distribution.

*Remark (Alternative: Empirical Estimation).* Instead of learnable parameters, one may estimate $P(Y|K)$ empirically via exponential moving average:

$$
\hat{P}(Y=y \mid K=k) = \frac{\text{EMA}[\mathbb{I}[Y=y, K=k]]}{\text{EMA}[\mathbb{I}[K=k]]}.

$$
This is non-differentiable w.r.t. chart assignment but more grounded in observations. A hybrid approach initializes learnable $\Theta$ from empirical estimates after warmup.

:::

:::{div} feynman-prose
Let me unpack this formula piece by piece.

The term $-\beta_{\text{class}} \log P(Y=y \mid K)$ is the semantic contribution. Remember that $-\log P$ is high when probability is low. So if chart $K$ strongly predicts class $y$ (high $P(Y=y|K)$), then $-\log P(Y=y|K)$ is low, which means $V_y$ is low there. Low potential means stable equilibrium---the system wants to stay there.

The coefficient $\beta_{\text{class}}$ controls how strongly class labels influence the dynamics. High $\beta_{\text{class}}$ means strong semantic gravity---the system really wants to find the right class region. Low $\beta_{\text{class}}$ means the base potential dominates and class labels have less influence.

Now, what's $P(Y=y|K)$? It's the probability of class $y$ given that we're in chart $K$. We have two ways to compute this:

1. **Learnable:** Keep a matrix $\Theta$ where $\Theta_{k,y}$ is a score for "how much does chart $k$ like class $y$." Apply softmax to get probabilities. This is differentiable and can be trained end-to-end.

2. **Empirical:** Just count what fraction of samples in chart $k$ have label $y$. This is more "honest" (it's what actually happens) but you can't backpropagate through it.

In practice, you might start empirical (to get a reasonable initialization) then switch to learnable (for fine-tuning).
:::

:::{admonition} Why Logarithms?
:class: feynman-added note

The logarithm in $-\log P(Y=y|K)$ isn't arbitrary. There are deep reasons why it's the right choice:

1. **Information-theoretic:** $-\log P$ is the surprisal or self-information. The semantic potential measures how "surprising" it is to find class $y$ in chart $K$. We want to minimize surprise.

2. **Additive composition:** If you have independent pieces of evidence, their log-probabilities add. Potentials add. The logarithm makes evidence composition work naturally with potential composition.

3. **Scale-appropriate gradients:** The gradient $\nabla(-\log P) = -\nabla P / P$ scales inversely with probability. This means rare classes get stronger gradient signal, which helps with class imbalance.

4. **Connection to free energy:** In statistical physics, $-\log P$ at inverse temperature $\beta_{\text{ent}}$ is exactly the free energy. The class temperature $\beta_{\text{class}}$ makes this connection explicit.
:::

:::{prf:definition} Region of Attraction
:label: def-region-of-attraction

The **region of attraction** for class $y$ is:

$$
\mathcal{B}_y := \{z \in \mathcal{Z} : \lim_{t \to \infty} \phi_t(z) \in \mathcal{A}_y\},

$$
where $\phi_t$ denotes the flow of the curl-corrected system

$$
\dot{z} = \mathcal{M}_{\text{curl}}\!\left(-G^{-1}(z)\nabla_A V_y(z)\right), \qquad \mathcal{M}_{\text{curl}} := (I - \beta_{\text{curl}} G^{-1}\mathcal{F})^{-1}
$$
(conservative case: $\mathcal{F}=0$).
Here $\nabla_A V_y := \nabla V_y - A$ with $A := \delta\Psi + \eta$ the non-conservative component of the reward 1-form
(conservative case: $A=0$).

*Interpretation:* $\mathcal{B}_y$ is the set of initial conditions from which the deterministic gradient flow on $V_y$ converges to the class-$y$ region.

:::

:::{div} feynman-prose
This is one of my favorite definitions because it turns classification into physics.

Imagine you drop a ball somewhere on the potential landscape $V_y$. The ball rolls downhill, following the gradient. Eventually it comes to rest at a local minimum. The region of attraction $\mathcal{B}_y$ is all the places you could drop the ball such that it ends up in the class-$y$ territory.

Notice the Riemannian metric $G^{-1}$ in the dynamics. The gradient isn't in Euclidean space---it respects the curved geometry of the latent manifold. If you're in a region where the metric is "stretched" (high $G$), you move more slowly. The geometry regulates the flow.

This is why I keep saying "classification is relaxation." You encode an input into latent space, then let it relax down the potential landscape. Where it settles is its classification. The math literally implements this: gradient descent on a potential until convergence.
:::

:::{prf:theorem} Classification as Relaxation
:label: thm-classification-as-relaxation

Under the overdamped dynamics ({ref}`sec-the-overdamped-limit`) with potential $V_y$:

$$
dz = \mathcal{M}_{\text{curl}}\!\left(-G^{-1}(z) \nabla_A V_y(z, K)\right) ds + \sqrt{2T_c}\, G^{-1/2}(z)\, dW_s, \quad T_c \text{ cognitive temperature } ({prf:ref}`def-cognitive-temperature`)

$$
When $\mathcal{F}=0$ (conservative case), $\mathcal{M}_{\text{curl}} = I$ and we recover pure gradient flow.
The limiting chart assignment satisfies:

$$
\lim_{s \to \infty} K(z(s)) \in \mathcal{A}_y \quad \text{almost surely},

$$
provided:
1. $z(0) \in \mathcal{B}_y$ (initial condition in the basin)
2. $T_c$ is sufficiently small (low temperature limit)
3. The basins have positive measure and are separated by finite barriers

*Proof sketch.* Define the Lyapunov function $L(z) := V_y(z, K(z))$ (see {cite}`khalil2002nonlinear` for Lyapunov theory, {cite}`lasalle1960invariance` for the invariance principle). Under the overdamped dynamics:

$$
\frac{dL}{ds} = \nabla_A V_y \cdot \dot{z} = -\nabla_A V_y \cdot \mathcal{M}_{\text{curl}} G^{-1}\nabla_A V_y + \text{noise terms}.

$$
The antisymmetric curl contribution in $\mathcal{M}_{\text{curl}}$ does no work, so it does not increase $L$.
For small $T_c$, the deterministic term dominates, ensuring $L$ decreases until $z$ reaches a local minimum. The class-$y$ region is the global minimum of $V_y$ by construction. Full proof in {ref}`sec-appendix-a-full-derivations`. $\square$

:::

:::{div} feynman-prose
There's something deeply satisfying about this theorem. It says: if you start in the right basin, and the temperature is low enough, and the barriers between classes are real, then the dynamics will find the right class.

The three conditions are worth thinking about:

1. **Starting in the basin:** This is the job of the encoder. The encoder must map inputs into latent positions that are at least in the right ballpark. The relaxation dynamics then refine this initial guess.

2. **Low temperature:** Temperature $T_c$ controls the noise level. High temperature means the system bounces around randomly and might hop over barriers into the wrong basin. Low temperature means deterministic descent dominates---you slide smoothly into your designated valley.

3. **Finite barriers:** The classes must actually be separated in the potential landscape. If two classes have no barrier between them, there's no stable way to distinguish them. This is a constraint on the learned representation.

The proof uses a beautiful piece of mathematics: Lyapunov theory. The potential itself acts as a Lyapunov function---it decreases along trajectories. The La Salle invariance principle tells us that the system converges to a set where the potential stops changing, which (for a well-designed $V_y$) is the class-$y$ minimum.
:::

:::{prf:corollary} Inference via Relaxation
:label: cor-inference-via-relaxation

Classification inference proceeds as:
1. Encode: $z_0 = \text{Enc}(x)$
2. Relax under neutral potential $V_{\text{base}}$ (no class conditioning) to equilibrium $z^*$
3. Read out: $\hat{y} = \arg\max_y P(Y=y \mid K(z^*))$

*Remark (Fast Path).* In practice, we often skip the relaxation and use direct readout: $\hat{y} = \arg\max_y \sum_k w_k(x) \cdot P(Y=y \mid K=k)$, where $w_k(x)$ are the router weights ({ref}`sec-tier-the-attentive-atlas`). The relaxation interpretation justifies this as the $T_c \to 0$, $s \to \infty$ limit.

**Cross-references:** {ref}`sec-the-overdamped-limit` (Overdamped Limit), Definition {prf:ref}`def-effective-potential`, {ref}`sec-the-hjb-correspondence` (Critic).

:::

:::{admonition} The Two Paths to Classification
:class: feynman-added example

Let me contrast the "proper" relaxation path with the fast path:

**The Relaxation Path (Theoretically Clean):**
```
Input x → Encode to z₀ → Let z flow: dz = M_curl(-∇V) → Wait for convergence → Read K(z*) → Predict argmax P(Y|K)
```
This is what the theorem describes. It's principled, it respects the geometry, and it has nice theoretical properties. But it requires running a dynamical system to convergence, which takes time.

**The Fast Path (Practically Useful):**
```
Input x → Compute router weights w(x) → Predict argmax Σₖ wₖ P(Y|K=k)
```
This skips the relaxation entirely. You just use the soft router weights to aggregate class predictions from each chart.

Why does the fast path work? The beautiful answer is that it's the *limit* of the relaxation path as $T_c \to 0$ and $s \to \infty$. In that limit, the system instantly snaps to the global minimum, and the router weights converge to indicators for the basin membership. The fast path implements this limiting behavior directly.

In practice, the fast path is good enough for inference. The relaxation picture matters more during training (it shapes what the representation learns) and for understanding edge cases (samples near basin boundaries might benefit from explicit relaxation).
:::

::::{admonition} Connection to RL #21: Imitation Learning as Degenerate Supervised Topology
:class: note
:name: conn-rl-21
**The General Law (Fragile Agent):**
Class labels define **Semantic Partitions** with **Class-Conditioned Potentials**:

$$
V_y(z, K) = -\beta_{\text{class}} \log P(Y=y \mid K) + V_{\text{base}}(z, K)

$$
Trajectories relax into class-specific basins via gradient flow on the learned metric.

**The Degenerate Limit:**
Set $V_{\text{base}} = 0$. Interpret labels as expert actions. Use Euclidean minimization.

**The Special Case (Standard RL):**

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}_{(s,a^*) \sim \mathcal{D}_{\text{expert}}}[-\log \pi(a^*|s)]

$$
This recovers **Behavioral Cloning** and **Imitation Learning** {cite}`pomerleau1991bc`.

**What the generalization offers:**
- **Metric segmentation:** Classes are metrically separated on the learned manifold
- **Potential landscape:** $V_y$ creates basins of attraction, not just classification boundaries
- **Jump modulation:** Cross-class transitions suppressed by separation penalty $\gamma_{\text{sep}}$
- **Relaxation dynamics:** Classification emerges from physics, not discrete argmax
::::

(sec-metric-segmentation-via-jump-rate-modulation)=
## Metric Segmentation via Jump Rate Modulation

:::{div} feynman-prose
Now we come to one of the cleverest ideas in this framework: enforcing class separation not by directly pushing embeddings apart, but by *making it hard to jump between class regions*.

Think about it like this. You could try to separate cats from dogs by pushing all the cat embeddings in one direction and all the dog embeddings in another. That's what contrastive learning does, and it works. But there's another approach: make the metric very "expensive" to traverse between cat territory and dog territory. You don't push them apart---you build a wall between them.

The wall isn't literal. It's implemented through the jump rates between charts. Remember, our latent space is tiled with charts, and transitions between charts happen via "jumps." The key insight is: **we can modulate how fast those jumps happen based on whether the source and destination charts belong to the same class**.

If both charts are cat charts, jump freely. If you're trying to jump from a cat chart to a dog chart, make it exponentially expensive. This effectively creates barriers between class regions without explicitly moving anything.
:::

:::{prf:definition} Class-Consistent Jump Rate
:label: def-class-consistent-jump-rate

For the WFR reaction term (Definition {prf:ref}`def-the-wfr-action`), modulate the inter-chart transition rate:

$$
\lambda_{i \to j}^{\text{sup}} := \lambda_{i \to j}^{(0)} \cdot \exp\left(-\gamma_{\text{sep}} \cdot D_{\text{class}}(i, j)\right),

$$
where:
- $\lambda^{(0)}_{i \to j}$ is the **base transition rate** from the GKSL master equation ({prf:ref}`def-gksl-generator`, {cite}`lindblad1976gksl,gorini1976gksl`, {ref}`sec-connection-to-gksl-master-equation`), derived from the overlap consistency of jump operators (Section 7.13)
- $\gamma_{\text{sep}} \geq 0$ is the **separation strength** (hyperparameter)
- $D_{\text{class}}(i, j) = \mathbb{I}[\text{Class}(i) \neq \text{Class}(j)]$ is the class disagreement indicator
- $\text{Class}(k) := \arg\max_y P(Y=y \mid K=k)$ is the dominant class of chart $k$

*Remark (Rate vs Operator).* {ref}`sec-factorized-jump-operators-efficient-chart-transitions` defines the **transition function** $L_{i \to j}$ (the coordinate change map). The **transition rate** $\lambda_{i \to j}$ is a separate quantity from the GKSL/master equation framework ({ref}`sec-connection-to-gksl-master-equation`, Equation 20.5.2) that governs *how often* jumps occur, not *where* they go. The rate is typically derived from the overlap structure: $\lambda_{i \to j}^{(0)} \propto \mathbb{E}_{x}[w_i(x) w_j(x)]$, measuring how much probability mass lies in the overlap $U_i \cap U_j$.

*Interpretation:* Transitions between charts of the same class proceed at the base rate $\lambda^{(0)}$. Transitions between charts of different classes are exponentially suppressed by factor $e^{-\gamma_{\text{sep}}}$.

:::

:::{div} feynman-prose
Let me walk through the formula. The base rate $\lambda^{(0)}_{i \to j}$ is determined by geometry---roughly, how much the charts overlap. Charts that share a lot of territory have high transition rates; charts that barely touch have low rates. This is the unsupervised part, determined by the manifold structure.

The class-modulation factor $\exp(-\gamma_{\text{sep}} \cdot D_{\text{class}})$ kicks in only when crossing class boundaries. The indicator $D_{\text{class}}(i,j)$ is 1 if charts $i$ and $j$ belong to different classes, 0 otherwise. So:

- **Same-class transition:** $D_{\text{class}} = 0$, factor is $e^0 = 1$, rate unchanged.
- **Cross-class transition:** $D_{\text{class}} = 1$, factor is $e^{-\gamma_{\text{sep}}}$, rate suppressed.

The hyperparameter $\gamma_{\text{sep}}$ controls how strong this suppression is. At $\gamma_{\text{sep}} = 0$, there's no suppression. At $\gamma_{\text{sep}} = 10$, cross-class jumps are $e^{-10} \approx 0.00005$ times as frequent as same-class jumps. That's serious suppression.
:::

:::{admonition} A Toll Road Between Cities
:class: feynman-added tip

Here's an analogy that might help. Imagine the latent space as a country with cities (charts) and roads between them. The base transition rates $\lambda^{(0)}$ are like the natural road connectivity---nearby cities have fast highways, distant cities require longer journeys.

Now suppose we want to separate two regions---say, Northern Territory (cats) and Southern Territory (dogs). We could physically move the cities apart, but that's disruptive. Instead, we build toll booths on every road crossing the border. The toll is $\gamma_{\text{sep}}$, and since time is money, the "effective distance" for crossing the border becomes much larger.

From a traveler's perspective, cities within the same territory are still easy to reach. But getting to the other territory requires paying the toll, which most travelers won't do unless they really need to.

The exponential form $e^{-\gamma_{\text{sep}}}$ comes from thinking about rates: if the toll makes crossings $e^{-\gamma_{\text{sep}}}$ times as frequent, it's as if the effective distance increased by $\gamma_{\text{sep}}$ (in log-rate terms).
:::

:::{prf:proposition} Effective Disconnection
:label: prop-effective-disconnection

As $\gamma_{\text{sep}} \to \infty$, the effective WFR distance between charts of different classes diverges:

$$
d_{\text{WFR}}(\mathcal{A}_{y_1}, \mathcal{A}_{y_2}) \to \infty \quad \text{for } y_1 \neq y_2.

$$
*Proof sketch.* The WFR distance (Definition {prf:ref}`def-the-wfr-action`) involves minimizing over paths that may use both transport (continuous flow within charts) and reaction (jumps between charts). Consider a path from $\mathcal{A}_{y_1}$ to $\mathcal{A}_{y_2}$:

1. **Transport-only paths:** If $\mathcal{A}_{y_1}$ and $\mathcal{A}_{y_2}$ are not geometrically adjacent (no shared chart boundary), pure transport paths have infinite cost.

2. **Jump paths:** Any path using cross-class jumps incurs reaction cost. In the GKSL interpretation ({ref}`sec-connection-to-gksl-master-equation`), the suppressed jump rate $\lambda^{\text{sup}} = \lambda^{(0)} e^{-\gamma_{\text{sep}}}$ means mass transfer between unlike-class charts requires longer dwell times, increasing the action.

3. **Divergence:** As $\gamma_{\text{sep}} \to \infty$, cross-class jumps become arbitrarily rare. The optimal path cost diverges because: (a) pure transport is blocked by chart boundaries, and (b) the reaction term penalizes staying in transition states waiting for rare jumps.

The precise scaling (exponential, polynomial, etc.) depends on the manifold geometry, but divergence is guaranteed. $\square$

:::

:::{div} feynman-prose
This proposition is saying something strong: crank up $\gamma_{\text{sep}}$ far enough and the classes become *metrically disconnected*. From the perspective of the WFR distance, cat territory and dog territory are infinitely far apart.

Why does this matter? Because distance in WFR geometry corresponds to how hard it is to transform one distribution into another. Infinite distance means it's impossible to smoothly morph a cat distribution into a dog distribution. The classes become topologically separated---you'd have to "teleport" to get from one to the other.

This is much stronger than just having a decision boundary. A decision boundary says "on this side it's cat, on that side it's dog." But samples near the boundary can still be similar in representation space. Metric disconnection says "there's no continuous path from cat to dog"---they live in different connected components of the effective geometry.
:::

:::{prf:remark} Tunneling as Anomaly Detection
:label: rem-tunneling-as-anomaly-detection

Cross-class transitions are not forbidden, merely exponentially suppressed. A detected cross-class jump indicates:

1. **Anomaly:** The sample lies in a transition region not well-covered by training
2. **Distribution shift:** The test distribution differs from training
3. **Adversarial input:** Deliberate perturbation to cross class boundaries

This provides a natural **out-of-distribution detection** mechanism: monitor the rate of cross-class transitions.

:::

:::{div} feynman-prose
This is a lovely bonus. We suppressed cross-class transitions for classification purposes, but we get an anomaly detector for free.

Here's the logic: if you trained the system properly, samples should stay in their class basins. A sample that *wants* to jump to another class is weird. Either it's genuinely ambiguous (edge case), or it doesn't fit the training distribution (anomaly), or someone is trying to fool you (adversarial attack).

By monitoring the "pressure" to make cross-class jumps, you get a signal about when inputs are strange. If $P(\text{cross-class jump}) \cdot \text{rate-suppression}$ starts rising, something's off.

This connects to a deep principle: good representations should be *stable* under the dynamics. Anomalies manifest as instabilities---they don't settle comfortably into any basin but keep trying to escape to other regions.
:::

:::{prf:definition} Class-Modulated Jump Operator
:label: def-class-modulated-jump-operator

Modify the jump operator (Definition {prf:ref}`def-factorized-jump-operator`) to incorporate class consistency:

```python
def class_modulated_jump_rate(
    lambda_base: torch.Tensor,    # [N_c, N_c] base jump rates
    chart_to_class: torch.Tensor, # [N_c, C] learnable logits
    gamma_sep: float = 5.0,       # Separation strength
) -> torch.Tensor:
    """
    Compute class-modulated jump rates.

    Cross-ref:
        - Definition 25.3.1 (Class-Consistent Jump Rate)
        - Definition 7.13.1 (Jump Operator)
    """
    # Get dominant class per chart
    p_y_given_k = F.softmax(chart_to_class, dim=1)  # [N_c, C]
    dominant_class = p_y_given_k.argmax(dim=1)       # [N_c]

    # Compute class disagreement matrix
    class_match = (dominant_class.unsqueeze(1) == dominant_class.unsqueeze(0)).float()  # [N_c, N_c]
    D_class = 1.0 - class_match  # 1 if classes differ, 0 if same

    # Modulate rates
    lambda_sup = lambda_base * torch.exp(-gamma_sep * D_class)

    return lambda_sup
```

**Cross-references:** {ref}`sec-the-wfr-metric` (WFR Metric), Definition {prf:ref}`def-factorized-jump-operator`, {ref}`sec-connection-to-gksl-master-equation` (GKSL Connection).

:::

(sec-the-supervised-topology-loss)=
## The Supervised Topology Loss

:::{div} feynman-prose
We define training losses that enforce the geometric structure described above.

So far we've described what we *want*: pure charts, separated classes, consistent routing. Now we need to turn those desires into loss functions that we can actually minimize.

The Supervised Topology Loss has four components, each enforcing a different aspect of the geometry. Think of it like a multi-objective fitness function: we want the representation to be good at predicting labels (route alignment), to have clean chart-class associations (purity), to use all its capacity (balance), and to keep different classes far apart (metric separation).

Let me take you through each one.
:::

(sec-chart-purity-loss)=
### Chart Purity Loss (Conditional Entropy)

:::{prf:definition} Purity Loss
:label: def-purity-loss

The purity loss measures how well charts separate classes:

$$
\mathcal{L}_{\text{purity}} = \sum_{k=1}^{N_c} P(K=k) \cdot H(Y \mid K=k),

$$
where:
- $P(K=k) = \mathbb{E}_{x \sim \mathcal{D}}[w_k(x)]$ is the marginal chart probability
- $H(Y \mid K=k) = -\sum_y P(Y=y \mid K=k) \log P(Y=y \mid K=k)$ is the class entropy within chart $k$

*Interpretation:* $\mathcal{L}_{\text{purity}} = H(Y \mid K)$, the conditional entropy of class given chart. Minimizing this encourages each chart to be associated with a single class.

:::

:::{div} feynman-prose
This loss asks: "If I tell you which chart a sample ended up in, how much uncertainty remains about its class?"

If charts are perfectly pure (each chart contains only one class), then $H(Y|K=k) = 0$ for all $k$, and the loss is zero. Knowing the chart tells you everything about the class.

If charts are maximally confused (each chart contains all classes equally), then $H(Y|K=k) = \log C$ (maximum entropy), and the loss is high. Knowing the chart tells you nothing about the class.

The weighting by $P(K=k)$ is important. A rarely-used chart contributes less to the loss than a frequently-used one. This makes sense: we care most about purifying the charts that actually get used.
:::

:::{prf:proposition} Purity-Information Duality
:label: prop-purity-information-duality

Minimizing $\mathcal{L}_{\text{purity}}$ is equivalent to maximizing the mutual information $I(K; Y)$:

$$
\mathcal{L}_{\text{purity}} = H(Y) - I(K; Y).

$$
Since $H(Y)$ is fixed by the data, $\min \mathcal{L}_{\text{purity}} \Leftrightarrow \max I(K; Y)$.

:::

:::{admonition} Why Mutual Information?
:class: feynman-added note

This proposition connects purity to information theory in a beautiful way.

Recall that $H(Y) = H(Y|K) + I(K;Y)$---total uncertainty equals conditional uncertainty plus mutual information. Since $H(Y)$ is determined by the class distribution in your dataset (you can't change it by learning), minimizing $H(Y|K)$ is exactly the same as maximizing $I(K;Y)$.

What is $I(K;Y)$? It's the amount of information that chart assignment carries about class labels. High mutual information means the charts are "informative" about classes---if you know the chart, you know a lot about the class.

So the purity loss is really asking: "Make the routing decision as informative as possible about the classification decision."
:::

(sec-load-balance-loss)=
### Load Balance Loss (Uniform Coverage)

{cite}`shazeer2017moe`

:::{prf:definition} Balance Loss
:label: def-balance-loss

Prevent degenerate solutions where all samples route to few charts:

$$
\mathcal{L}_{\text{balance}} = D_{\text{KL}}\left(\bar{w} \;\|\; \text{Uniform}(N_c)\right),

$$
where $\bar{w} = \mathbb{E}_{x \sim \mathcal{D}}[w(x)]$ is the average router weight vector.

*Interpretation:* Encourages all charts to be used, preventing "dead charts" and ensuring the atlas covers the label space.

:::

:::{div} feynman-prose
This is the "anti-collapse" loss. Without it, the system might find that it's easiest to just use one or two charts for everything. That's technically a valid solution, but it wastes all the representational capacity you built into your model.

The balance loss measures how far the average routing distribution is from uniform. If every chart gets used equally often, $\bar{w} = (1/N_c, \ldots, 1/N_c)$ and the KL divergence is zero. If all mass concentrates on one chart, the KL divergence blows up.

This is a well-known trick from the mixture-of-experts literature. Shazeer et al. found that without load balancing, expert networks collapse to using just a handful of experts. The same principle applies here: charts are like experts, and we need to encourage the system to use all of them.
:::

:::{admonition} The Expert Collapse Problem
:class: feynman-added warning

Why does collapse happen without balancing? Here's the intuition:

Early in training, some charts will be randomly better than others for some subset of the data. The router learns to send that data to those charts. But now those charts see more data and get even better. Meanwhile, the neglected charts see less data and stagnate. This positive feedback loop leads to "rich get richer" dynamics where a few charts dominate and the rest become useless.

The balance loss breaks this feedback. It says: "Yes, you can route to whichever chart works best, but I'm going to penalize you for being too uneven." This forces the system to find ways to use all charts productively.

The trade-off controlled by $\lambda_{\text{bal}}$ (typically small, like 0.01) is between routing quality and routing diversity. Too much balancing and you force bad routings; too little and you collapse.
:::

(sec-metric-contrastive-loss)=
### Metric Contrastive Loss (Geometric Separation)

{cite}`schroff2015facenet,khosla2020supcon`

:::{prf:definition} Contrastive Loss
:label: def-contrastive-loss

Enforce that different-class samples are geometrically separated:

$$
\mathcal{L}_{\text{metric}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}: y_i \neq y_j} w_i^\top w_j \cdot \max(0, m - d_{\text{jump}}(z_i, z_j))^2,

$$
where:
- $\mathcal{P}$ is the set of sample pairs in the batch
- $w_i, w_j$ are router weight vectors
- $m > 0$ is the margin (minimum desired separation)
- $d_{\text{jump}}(z_i, z_j)$ is the minimum jump cost ({ref}`sec-factorized-jump-operators-efficient-chart-transitions`)

*Interpretation:* If two samples have different labels but high router overlap ($w_i^\top w_j$ large), they must be separated by at least margin $m$ in jump distance. Otherwise, the loss penalizes the configuration.

:::

:::{div} feynman-prose
This loss says: "Different classes should be geometrically far apart."

The structure is classic contrastive learning, but with a twist. The term $w_i^\top w_j$ measures how much two samples share routing. If they route through completely different charts, $w_i^\top w_j \approx 0$ and there's no penalty regardless of their distance. But if they route through similar charts (high overlap), then we demand that they be far apart if they have different labels.

The hinge form $\max(0, m - d)^2$ is a margin loss: distances greater than $m$ contribute zero loss, while distances smaller than $m$ contribute a penalty that grows quadratically as you get closer.

The margin $m$ is a hyperparameter that sets your desired minimum separation. Set it too high and you're asking for more separation than the geometry can provide. Set it too low and classes can still be confusably close.
:::

:::{admonition} The Role of Router Overlap
:class: feynman-added tip

The factor $w_i^\top w_j$ is subtle but important. Why weight the contrastive penalty by routing similarity?

The answer is efficiency. Most pairs of different-class samples are *already* separated just by routing to different charts. A cat that routes 90% to chart 3 and a dog that routes 90% to chart 7 have low overlap ($w_i^\top w_j \approx 0.1$) and don't need explicit pushing apart---the routing already separates them.

The pairs we care about are those with high routing overlap despite being different classes. These are the boundary cases, the hard examples, the potential confusions. By weighting by overlap, the loss focuses its effort on exactly the pairs that need work.

This is much more efficient than comparing all pairs equally. With $N$ samples per batch, there are $O(N^2)$ pairs. Most of them are easy. The overlap weighting lets us focus on the $O(N)$ hard ones.
:::

(sec-route-alignment-loss)=
### Route Alignment Loss (Prediction Consistency)

:::{prf:definition} Route Alignment Loss
:label: def-route-alignment-loss

The primary classification loss:

$$
\mathcal{L}_{\text{route}} = \mathbb{E}_{x, y_{\text{true}}}\left[\text{CE}\left(\sum_k w_k(x) \cdot P(Y=\cdot \mid K=k), \; y_{\text{true}}\right)\right],

$$
where $\text{CE}$ denotes cross-entropy.

*Interpretation:* The predicted class distribution is the router-weighted average of per-chart class distributions. This must match the true label.

:::

:::{div} feynman-prose
This is the "make correct predictions" loss---the supervised learning objective we're all familiar with, just written in our chart-based language.

Here's how prediction works: each chart $k$ has its own class distribution $P(Y|K=k)$. The router gives us weights $w_k(x)$ saying how much we trust each chart for this particular input. The final prediction is the weighted average: $\sum_k w_k(x) \cdot P(Y|K=k)$.

The route alignment loss is just cross-entropy between this prediction and the true label. If the prediction is confident and correct, loss is low. If it's confident and wrong, or uncertain about the right answer, loss is high.

This is the loss that actually teaches the system to classify. The other losses (purity, balance, metric) shape the geometry, but this one provides the supervised signal.
:::

(sec-combined-supervised-topology-loss)=
### Combined Supervised Topology Loss

:::{prf:definition} Total Loss
:label: def-total-loss

The full supervised topology loss:

$$
\mathcal{L}_{\text{sup-topo}} = \mathcal{L}_{\text{route}} + \lambda_{\text{pur}} \mathcal{L}_{\text{purity}} + \lambda_{\text{bal}} \mathcal{L}_{\text{balance}} + \lambda_{\text{met}} \mathcal{L}_{\text{metric}}.

$$
Typical hyperparameters: $\lambda_{\text{pur}} = 0.1$, $\lambda_{\text{bal}} = 0.01$, $\lambda_{\text{met}} = 0.01$.

**Algorithm 25.4.7 (SupervisedTopologyLoss Implementation).**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SupervisedTopologyLoss(nn.Module):
    """
    Supervised topology loss enforcing chart purity, balance, and separation.

    Cross-ref:
        - Definition 25.4.6 (Total Loss)
        - {ref}`sec-tier-the-attentive-atlas` (Router Weights)
    """

    def __init__(
        self,
        num_charts: int,
        num_classes: int,
        lambda_purity: float = 0.1,
        lambda_balance: float = 0.01,
        lambda_metric: float = 0.01,
        margin: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_charts = num_charts
        self.num_classes = num_classes
        self.lambda_purity = lambda_purity
        self.lambda_balance = lambda_balance
        self.lambda_metric = lambda_metric
        self.margin = margin

        # Learnable chart-to-class mapping (Definition 25.2.1)
        self.chart_to_class = nn.Parameter(
            torch.randn(num_charts, num_classes) * 0.01
        )
        self.temperature = temperature

    @property
    def p_y_given_k(self) -> torch.Tensor:
        """P(Y|K) distribution [N_c, C]."""
        return F.softmax(self.chart_to_class / self.temperature, dim=1)

    def forward(
        self,
        router_weights: torch.Tensor,  # [B, N_c]
        y_true: torch.Tensor,          # [B] class labels
        z_latent: torch.Tensor = None, # [B, D] optional for metric loss
    ) -> Dict[str, torch.Tensor]:
        """
        Compute supervised topology losses.

        Returns dict with individual losses and total.
        """
        B = router_weights.shape[0]
        p_y_k = self.p_y_given_k  # [N_c, C]

        # === Route Alignment Loss (Definition 25.4.5) ===
        # P(Y|x) = sum_k w_k(x) * P(Y|K=k)
        p_y_x = torch.matmul(router_weights, p_y_k)  # [B, C]
        loss_route = F.cross_entropy(
            torch.log(p_y_x + 1e-8), y_true
        )

        # === Purity Loss (Definition 25.4.1) ===
        # H(Y|K=k) for each chart
        entropy_per_chart = -(p_y_k * torch.log(p_y_k + 1e-8)).sum(dim=1)  # [N_c]
        # P(K=k) = average router weight
        p_k = router_weights.mean(dim=0)  # [N_c]
        # L_purity = sum_k P(K=k) * H(Y|K=k)
        loss_purity = (p_k * entropy_per_chart).sum()

        # === Balance Loss (Definition 25.4.3) ===
        # KL(p_k || Uniform) = sum_k p_k * log(p_k / (1/N_c)) = sum_k p_k * (log(p_k) + log(N_c))
        uniform = torch.ones_like(p_k) / self.num_charts
        # Manual KL computation: KL(P||Q) = sum P * log(P/Q)
        loss_balance = (p_k * (torch.log(p_k + 1e-8) - torch.log(uniform))).sum()

        # === Metric Contrastive Loss (Definition 25.4.4) ===
        loss_metric = torch.tensor(0.0, device=router_weights.device)
        if self.lambda_metric > 0 and B > 1:
            # Router overlap as proxy for proximity
            # w_i^T w_j measures routing similarity
            overlap = torch.matmul(router_weights, router_weights.t())  # [B, B]

            # Class disagreement mask
            y_match = (y_true.unsqueeze(1) == y_true.unsqueeze(0)).float()
            y_diff = 1.0 - y_match  # 1 if different classes

            # Penalize high overlap for different-class pairs
            # Using overlap as proxy for d_jump (lower overlap ~ larger distance)
            pseudo_dist = 1.0 - overlap  # Rough proxy
            hinge = F.relu(self.margin - pseudo_dist)
            loss_metric = (y_diff * overlap * hinge ** 2).sum() / (y_diff.sum() + 1e-8)

        # === Total Loss ===
        loss_total = (
            loss_route
            + self.lambda_purity * loss_purity
            + self.lambda_balance * loss_balance
            + self.lambda_metric * loss_metric
        )

        return {
            'loss_total': loss_total,
            'loss_route': loss_route,
            'loss_purity': loss_purity,
            'loss_balance': loss_balance,
            'loss_metric': loss_metric,
        }
```

**Cross-references:** {ref}`sec-tier-the-attentive-atlas` (Router Weights), Section 7.13 (Jump Operators), {ref}`sec-diagnostics-stability-checks` (Diagnostic Nodes).

:::

:::{div} feynman-prose
The total loss is just a weighted sum of the four components. The weights $\lambda$ control the relative importance of each objective.

Let me give you some intuition for the typical values:

- **$\lambda_{\text{pur}} = 0.1$:** Purity is important but not dominant. We want charts to specialize, but not at the expense of prediction accuracy. If you set this too high, charts become so specialized they can't generalize.

- **$\lambda_{\text{bal}} = 0.01$:** Balance is a soft constraint. We want to avoid collapse but not force unnatural uniformity. A small weight nudges toward balance without fighting the natural structure of the data.

- **$\lambda_{\text{met}} = 0.01$:** The metric loss is supplementary. The primary separation comes from jump rate modulation and purity. The metric loss handles edge cases where different-class samples end up in overlapping chart regions.

In practice, you might tune these on a validation set. But the default values are reasonable starting points for most problems.
:::

:::{admonition} Anatomy of the Implementation
:class: feynman-added example

Let me walk through the code step by step, matching each piece to the math.

**The learnable chart-to-class mapping:**
```python
self.chart_to_class = nn.Parameter(torch.randn(num_charts, num_classes) * 0.01)
```
This is $\Theta \in \mathbb{R}^{N_c \times C}$ from Definition {prf:ref}`def-class-conditioned-potential`. Each entry $\Theta_{k,y}$ is the "affinity" of chart $k$ for class $y$. We initialize it near zero (small random values) so all charts start with roughly equal preference for all classes.

**Computing P(Y|K):**
```python
return F.softmax(self.chart_to_class / self.temperature, dim=1)
```
Apply softmax along the class dimension. Temperature controls sharpness: low temperature gives more peaked distributions (charts commit strongly to one class), high temperature gives flatter distributions (charts remain agnostic).

**Route alignment loss:**
```python
p_y_x = torch.matmul(router_weights, p_y_k)  # [B, C]
loss_route = F.cross_entropy(torch.log(p_y_x + 1e-8), y_true)
```
Matrix multiply router weights $[B, N_c]$ with class distributions $[N_c, C]$ to get batch predictions $[B, C]$. Then compute cross-entropy against true labels.

**Purity loss:**
```python
entropy_per_chart = -(p_y_k * torch.log(p_y_k + 1e-8)).sum(dim=1)
p_k = router_weights.mean(dim=0)
loss_purity = (p_k * entropy_per_chart).sum()
```
Compute entropy of each chart's class distribution, weight by how often each chart is used, sum. This is exactly $H(Y|K)$.

**Balance loss:**
```python
loss_balance = (p_k * (torch.log(p_k + 1e-8) - torch.log(uniform))).sum()
```
This is $D_{KL}(\bar{w} \| \text{Uniform})$, computed directly from the definition of KL divergence.

**Metric loss:**
The implementation uses router overlap as a proxy for distance, which is a simplification of the full jump-distance computation. It focuses computational effort on pairs that share routing (high overlap) but have different classes.
:::

(sec-thermodynamics-conditioned-generation)=
## Thermodynamics: Conditioned Generation

:::{div} feynman-prose
This framework unifies classification with the generative law ({ref}`sec-radial-generation-entropic-drift-and-policy-control`).

Now here's where things get really elegant. We've been treating class labels as things we *predict*---given an input, which class? But we can also treat class labels as things we *condition on*---given a class, generate an input.

The semantic potential $V_y$ works both ways. For classification, it defines basins that samples relax into. For generation, it defines the landscape we sample from. Same potential, two uses.

This is the hallmark of a good theoretical framework: the same machinery explains multiple phenomena. Classification and generation aren't separate problems---they're two directions through the same geometric structure.
:::

:::{prf:remark} Connection to Mobius Re-centering
:label: rem-connection-to-m-bius-re-centering

The Mobius re-centering $\phi_c$ for conditioned generation (Definition {prf:ref}`ax-bulk-boundary-decoupling`) can be interpreted as centering at the **class centroid**:

$$
c_y := \mathbb{E}_{x: Y(x)=y}[\text{Enc}(x)],

$$
i.e., the average latent position of class-$y$ samples. Conditioned generation "starts" the holographic expansion from this centroid.

:::

:::{div} feynman-prose
The class centroid is the "center of mass" of all samples with label $y$ in the latent space. When we do class-conditioned generation, we begin the generative process not at the origin but at this centroid.

Think of it like this: if you want to generate a cat, you start at the "average cat location" in latent space and then add variation around it. The Mobius transformation effectively re-centers the coordinate system on the cat centroid, so that generation proceeds outward from there.

This is a form of "conditioning by re-centering"---changing where you stand in the space rather than changing the dynamics themselves.
:::

:::{prf:proposition} Class-Conditioned Langevin
:label: prop-class-conditioned-langevin

The generative Langevin equation {cite}`welling2011sgld,song2019ncsn` (Definition {prf:ref}`prop-so-d-symmetry-at-origin`) with class conditioning becomes:

$$
dz = \mathcal{M}_{\text{curl}}\!\left(-\nabla_G V_y(z, K)\right) d\tau + \sqrt{2T_c}\,G^{-1/2}(z)\,dW_\tau,

$$
where $V_y$ is the class-conditioned potential (Definition {prf:ref}`def-class-conditioned-potential`).

*Interpretation:* To generate a sample of class $y$, we run Langevin dynamics with the $V_y$ potential. The semantic term $-\beta_{\text{class}} \log P(Y=y \mid K)$ biases the flow toward class-$y$ charts.

:::

:::{div} feynman-prose
Langevin dynamics is one of the most beautiful tools in all of computational physics. You have a potential landscape, you start somewhere, and you let the system roll downhill while being buffeted by random noise. Eventually you sample from the equilibrium distribution $\propto e^{-V/T}$.

For class-conditioned generation, we use $V_y$ as the potential. This potential has two parts: the base potential $V_{\text{base}}$ (which gives realistic outputs) and the semantic term $-\beta_{\text{class}} \log P(Y=y|K)$ (which biases toward class $y$).

The result: the Langevin process prefers to spend time in regions where $V_y$ is low. That means regions where both: (a) outputs look realistic (low $V_{\text{base}}$), and (b) the class is $y$ (high $P(Y=y|K)$). Exactly what we want for class-conditioned generation.
:::

:::{prf:corollary} Label as Symmetry-Breaking Field, cf. classifier-free guidance {cite}`ho2022cfg`
:label: cor-label-as-symmetry-breaking-field-cf-classifier-free-guidance

The class label $y$ breaks the $SO(2)$ symmetry of the unconditioned flow in the Poincare disk. At the origin:

1. **Unconditioned:** $\nabla_A V_{\text{base}}(0) = 0$ (symmetric saddle)
2. **Conditioned:** $\nabla_A V_y(0) = -\beta_{\text{class}} \nabla_z \log P(Y=y \mid K(z))|_{z=0} \neq 0$

The non-zero gradient aligns the initial "kick" direction with the class-$y$ basin.

:::

:::{div} feynman-prose
This is physics language, but the idea is simple. At the origin of the latent space (before any "choice" has been made), the unconditioned dynamics are symmetric---there's no preferred direction. Every class basin is equally accessible.

The class label breaks this symmetry. It adds a term to the potential that points toward the class-$y$ region. Now there *is* a preferred direction: toward wherever class $y$ lives.

This is exactly what classifier-free guidance does in diffusion models: you compute both conditioned and unconditioned scores, and use the conditioned one to bias the generation. Here we see why that works---the class label creates an asymmetry in the potential landscape that guides the generative flow.

The strength of this guidance is controlled by $\beta_{\text{class}}$. High $\beta_{\text{class}}$ means strong guidance (stay very close to class $y$). Low $\beta_{\text{class}}$ means weak guidance (explore more, even into other class regions).
:::

:::{prf:definition} Class Centroid in Poincare Disk
:label: def-class-centroid-in-poincar-disk

For the Poincare disk embedding {cite}`nickel2017poincare,ganea2018hnn`, define the class centroid using the **Frechet mean** {cite}`lou2020frechet`:

$$
c_y := \arg\min_{c \in \mathbb{D}} \sum_{x: Y(x)=y} d_{\mathbb{D}}(c, \text{Enc}(x))^2.

$$
This is well-defined since the Poincare disk has negative curvature (unique Frechet means).

**Cross-references:** {ref}`sec-policy-control-field` (Langevin Dynamics), {ref}`sec-the-retrieval-texture-firewall` (Mobius Re-centering), Definition {prf:ref}`prop-so-d-symmetry-at-origin`.

:::

:::{admonition} Why Hyperbolic Centroids Are Special
:class: feynman-added note

In Euclidean space, the centroid (mean) of a set of points is straightforward: just average the coordinates. In hyperbolic space, like the Poincare disk, it's more subtle.

The **Frechet mean** is the generalization of the centroid to curved spaces: it's the point that minimizes the sum of squared distances to all the data points. In Euclidean space, this gives you the ordinary mean. In hyperbolic space, it gives you something different.

Why does this matter? Hyperbolic space has the property that volume grows exponentially with radius. This means class centroids tend to sit *closer to the origin* than you might expect from Euclidean intuition. Classes spread out toward the boundary, but their "centers of mass" cluster toward the middle.

The Frechet mean also has the nice property of being unique in negative-curvature spaces (like the Poincare disk). In positive-curvature spaces, you can have multiple local minima, but hyperbolic space is well-behaved.
:::

:::{prf:remark} Integration with TopologicalDecoder
:label: rem-integration-with-topologicaldecoder

The TopologicalDecoder ({ref}`sec-decoder-architecture-overview-topological-decoder`) receives the geometric content $z_{\text{geo}} = e_K + z_n$ and routes through chart-specific projectors. For class-conditioned generation:

1. **Class determines charts:** The class label $y$ biases chart selection toward $\mathcal{A}_y$ via the semantic potential $V_y$
2. **Decoder routing:** The TopologicalDecoder's inverse router ({ref}`sec-topological-decoder-module`) can either:
   - Accept an explicit chart index $K$ (from the generative flow)
   - Infer routing from $z_{\text{geo}}$ (autonomous mode)
3. **Consistency constraint:** The decoder's inferred routing should agree with the encoder's class-conditioned routing:

   $$
   \mathcal{L}_{\text{route-consistency}} = \mathbb{E}_{x,y}\left[\text{CE}\left(w_{\text{dec}}(z_{\text{geo}}), w_{\text{enc}}(x)\right)\right]

   $$
   where $w_{\text{dec}}$ are the decoder's soft router weights and $w_{\text{enc}}$ are the encoder's.

This ensures that class-conditioned generation produces samples that the encoder would classify correctly---a form of **cycle consistency** between encoding and decoding under the semantic topology.

:::

:::{div} feynman-prose
This remark connects our classification framework to the decoder. The key insight is cycle consistency.

Here's the scenario: we generate a sample conditioned on class $y$. The generative flow goes through certain charts (biased by $V_y$ toward class-$y$ charts). Now we take that generated sample and encode it back. Does the encoder agree that it's class $y$?

If the system is consistent, yes. The generated sample lives in class-$y$ territory by construction (we biased the generation that way). The encoder, seeing this sample, should route it back to the same territory and classify it as $y$.

The route-consistency loss enforces this. It says: "The decoder's routing decisions should match what the encoder would do." This closes the loop and ensures that generation and classification are genuinely inverse operations.

Without this consistency, you could have a system that generates samples the encoder doesn't recognize, or classifies samples that the decoder can't reconstruct. The cycle consistency keeps everything aligned.
:::

(sec-hierarchical-classification-via-scale-decomposition)=
## Hierarchical Classification via Scale Decomposition

:::{div} feynman-prose
Real-world categories are hierarchical (e.g., Animal -> Dog -> Terrier). The **stacked TopoEncoder** ({ref}`sec-stacked-topoencoders-deep-renormalization-group-flow`) naturally reflects this.

Most classification problems have implicit hierarchy. A Terrier is a Dog is an Animal is a Living Thing. ImageNet has 1000 leaf classes but they cluster into broader categories (vehicles, animals, furniture, etc.).

Standard classification treats all classes as equally unrelated. But that's clearly wrong---misclassifying a Terrier as a Poodle is less bad than misclassifying it as a Truck. The semantic structure matters.

Our framework captures this naturally. The stacked TopoEncoder operates at multiple scales: coarse (bulk) layers capture broad categories, fine (boundary) layers capture detailed distinctions. By aligning label hierarchy with scale hierarchy, we get a classifier that "thinks" hierarchically.
:::

:::{prf:definition} Hierarchical Labels
:label: def-hierarchical-labels

A **label hierarchy** is a sequence of label spaces:

$$
\mathcal{Y}_0 \twoheadrightarrow \mathcal{Y}_1 \twoheadrightarrow \cdots \twoheadrightarrow \mathcal{Y}_L,

$$
where $\twoheadrightarrow$ denotes a surjection (coarsening). $\mathcal{Y}_0$ are coarse labels (super-categories), $\mathcal{Y}_L$ are fine labels (leaf categories).

*Example:* $\mathcal{Y}_0 = \{\text{Animal}, \text{Vehicle}\}$, $\mathcal{Y}_1 = \{\text{Dog}, \text{Cat}, \text{Car}, \text{Bike}\}$, $\mathcal{Y}_2 = \{\text{Terrier}, \text{Poodle}, \ldots\}$.

:::

:::{div} feynman-prose
The surjection arrows $\twoheadrightarrow$ mean "can be coarsened to." Each fine label maps to exactly one coarser label. Terrier maps to Dog maps to Animal. This forms a tree structure.

The label hierarchy tells us how to group classes at different levels of abstraction. At the coarsest level (level 0), you just distinguish Animal from Vehicle. At the finest level (level $L$), you distinguish all the leaf categories.
:::

:::{prf:proposition} Scale-Label Alignment
:label: prop-scale-label-alignment

In the stacked TopoEncoder ({ref}`sec-stacked-topoencoders-deep-renormalization-group-flow`), enforce purity at each scale:

- **Layer 0 (Bulk/Slow):** Charts at level 0 correspond to coarse classes. Enforce:

  $$
  \mathcal{L}_{\text{purity}}^{(0)} = H(\mathcal{Y}_0 \mid K^{(0)})

  $$
- **Layer $\ell$ (Intermediate):** Charts at level $\ell$ correspond to level-$\ell$ classes. Enforce:

  $$
  \mathcal{L}_{\text{purity}}^{(\ell)} = H(\mathcal{Y}_\ell \mid K^{(\ell)})

  $$
- **Layer $L$ (Boundary/Fast):** Charts at level $L$ correspond to fine classes. Enforce:

  $$
  \mathcal{L}_{\text{purity}}^{(L)} = H(\mathcal{Y}_L \mid K^{(L)})

  $$
:::

:::{div} feynman-prose
The idea is beautiful in its simplicity: match the scale of the label to the scale of the representation.

At the bulk (deep, slow) layers of the network, we don't expect to distinguish Terriers from Poodles. These fine distinctions require subtle features that only emerge at the boundary (shallow, fast) layers. But bulk layers *can* distinguish Animals from Vehicles---that's a coarse distinction that shows up even in low-resolution features.

So we enforce hierarchy: bulk charts should be pure with respect to coarse labels, boundary charts should be pure with respect to fine labels. The intermediate layers handle intermediate labels.

This alignment isn't just aesthetic---it's computationally efficient. It means the network doesn't try to do fine classification at layers where it doesn't have the representational power for it. Instead, it builds up the classification hierarchically, making coarse decisions early and refining them as it goes.
:::

:::{prf:remark} Renormalization Group Interpretation
:label: rem-renormalization-group-interpretation

The semantic hierarchy matches the physical renormalization scale:

| Scale                | Latent Structure              | Semantic Structure |
|----------------------|-------------------------------|--------------------|
| Bulk (Layer 0)       | Slow modes, large wavelengths | Super-categories   |
| Intermediate         | Medium modes                  | Categories         |
| Boundary (Layer $L$) | Fast modes, fine details      | Sub-categories     |

This is the **semantic RG flow**: coarse-graining in the label space corresponds to flowing toward the bulk in latent space.

:::

:::{admonition} What is the Renormalization Group?
:class: feynman-added tip

The Renormalization Group (RG) is one of the deepest ideas in physics. It describes how physical systems look different at different scales.

Imagine zooming out from a picture. At high resolution, you see fine details: individual pixels, textures, edges. As you zoom out, details blur together. What remains are the large-scale structures: shapes, colors, overall composition.

The RG says this isn't just losing information---it's *systematically* losing the *right* information. At each scale, certain patterns dominate and others become irrelevant. The "flow" from fine to coarse follows predictable rules.

In our semantic context: fine categories (Terrier, Poodle) are "high-resolution" features that blur into coarser categories (Dog, Cat) as you zoom out. At the coarsest scale, you just see "Animal." This semantic coarse-graining is the same mathematical structure as physical coarse-graining---that's the RG connection.

The stacked TopoEncoder literally implements this: bulk layers see coarse features, boundary layers see fine features, and the flow from boundary to bulk is the semantic RG flow.
:::

:::{prf:definition} Hierarchical Supervised Loss
:label: def-hierarchical-supervised-loss

The total hierarchical loss:

$$
\mathcal{L}_{\text{hier}} = \sum_{\ell=0}^{L} \alpha_\ell \left(\mathcal{L}_{\text{route}}^{(\ell)} + \lambda_{\text{pur}} \mathcal{L}_{\text{purity}}^{(\ell)}\right),

$$
where $\alpha_\ell$ weights the contribution of each scale (typically $\alpha_\ell = 1$ or decaying with $\ell$).

**Cross-references:** {ref}`sec-stacked-topoencoders-deep-renormalization-group-flow` (Stacked TopoEncoder), Definition {prf:ref}`def-the-peeling-step`, {ref}`sec-rigorous-interpretation-renormalization-group-flow` (RG Interpretation).

:::

:::{div} feynman-prose
The hierarchical loss sums the route alignment and purity losses across all scales, weighted by $\alpha_\ell$.

Why might you use decaying weights $\alpha_\ell$? One reason: coarse decisions are more important than fine ones. Getting Animal vs Vehicle right is more crucial than getting Terrier vs Poodle right. Decaying weights (higher $\alpha$ for coarse scales) encode this priority.

Another reason: fine distinctions are harder to learn. Giving them equal weight can lead to the network spending all its effort on subtle distinctions while ignoring obvious ones. Decaying weights ensure the basics are learned first.

In practice, equal weights ($\alpha_\ell = 1$) often work fine, especially with good initialization. The hierarchical structure is more important than the exact weighting.
:::

(sec-summary-and-diagnostic-nodes)=
## Summary and Diagnostic Nodes

:::{div} feynman-prose
Let's take stock of what we've built. We started with a simple question---what does classification mean geometrically?---and constructed a complete framework:

1. **Semantic Partitions:** Class labels induce a soft partition of the chart atlas, grouping charts by their dominant class.

2. **Class-Conditioned Potentials:** Each class $y$ has a potential $V_y$ that creates basins of attraction in latent space.

3. **Jump Rate Modulation:** Cross-class transitions are suppressed, effectively disconnecting different classes in the metric.

4. **Multi-scale Losses:** Purity, balance, route alignment, and metric separation work together to train a geometrically coherent classifier.

5. **Hierarchical Extension:** The framework naturally extends to hierarchical labels via the stacked encoder's scale decomposition.

The result is classification that's not just accurate but *geometric*---classes are regions, boundaries are barriers, and prediction is relaxation.
:::

**Table 25.7.1 (Summary of Supervised Topology Laws).**

:::{div} feynman-added
| Aspect             | Formula                                                                                   | Units       | Reference                                      |
|--------------------|-------------------------------------------------------------------------------------------|-------------|------------------------------------------------|
| Semantic Partition | $\mathcal{A}_y = \{k: P(Y=y\mid K=k) > 1-\epsilon\}$                                      | ---           | Def {prf:ref}`def-semantic-partition`          |
| Class Potential    | $V_y = -\beta_{\text{class}} \log P(Y=y\mid K) + V_{\text{base}}$                         | nat         | Def {prf:ref}`def-class-conditioned-potential` |
| Jump Modulation    | $\lambda_{i\to j}^{\text{sup}} = \lambda^{(0)} e^{-\gamma_{\text{sep}} D_{\text{class}}}$ | step$^{-1}$ | Def {prf:ref}`def-class-consistent-jump-rate`  |
| Purity Loss        | $\sum_k P(K=k) H(Y\mid K=k)$                                                              | nat         | Def {prf:ref}`def-purity-loss`                 |
| Route Alignment    | $\text{CE}(\sum_k w_k P(Y\mid K=k), y_{\text{true}})$                                     | nat         | Def {prf:ref}`def-route-alignment-loss`        |
:::

(node-40)=
**Node 40: PurityCheck (CapacitySaturationCheck)**

:::{div} feynman-prose
Following the diagnostic node convention ({ref}`sec-theory-thin-interfaces`), we define checks that monitor the health of the supervised topology.
:::

| **#**  | **Name**        | **Component** | **Type**                | **Interpretation**     | **Proxy**     | **Cost** |
|--------|-----------------|---------------|-------------------------|------------------------|---------------|----------|
| **40** | **PurityCheck** | **Router**    | **Semantic Clustering** | Are charts class-pure? | $H(Y \mid K)$ | $O(BC)$  |

**Trigger conditions:**
- High PurityCheck: Charts contain mixed classes; classification boundaries fall within charts.
- Remedy: Increase purity loss weight $\lambda_{\text{pur}}$; increase number of charts; check for insufficient training data per class.

:::{admonition} Interpreting Purity Diagnostics
:class: feynman-added note

What does it mean when purity is high (bad)?

**Diagnosis 1: Too few charts.** If you have 10 classes and only 5 charts, at least some charts must handle multiple classes. Solution: add more charts.

**Diagnosis 2: Classes genuinely overlap.** Some classification problems have classes that are intrinsically hard to separate. In this case, high purity might be unavoidable, and you should focus on good routing rather than chart purity.

**Diagnosis 3: Training issues.** Maybe you have enough charts, but training hasn't converged. Check learning curves. The purity loss should decrease over training.

**Diagnosis 4: Class imbalance.** If one class dominates, all charts might end up associated with it. Check your class distribution and consider balancing.
:::

(node-41)=
**Node 41: ClassSeparationCheck (SupervisedTopologyChecks)**

| **#**  | **Name**                 | **Component** | **Type**             | **Interpretation**                          | **Proxy**                                                                  | **Cost**     |
|--------|--------------------------|---------------|----------------------|---------------------------------------------|----------------------------------------------------------------------------|--------------|
| **41** | **ClassSeparationCheck** | **Jump Op**   | **Class Separation** | Are different classes metrically separated? | $\min_{y_1 \neq y_2} d_{\text{WFR}}(\mathcal{A}_{y_1}, \mathcal{A}_{y_2})$ | $O(C^2 N_c)$ |

**Trigger conditions:**
- Low ClassSeparationCheck: Different classes are metrically close; cross-class transitions are too frequent.
- Remedy: Increase separation strength $\gamma_{\text{sep}}$; add metric contrastive loss; check for class imbalance.

:::{div} feynman-prose
The separation check monitors the minimum distance between any two class regions. If this is low, the classes are dangerously close---confusions are likely.

The proxy (WFR distance between sub-atlases) is expensive to compute exactly, so in practice you might use a cheaper approximation based on router overlap statistics.
:::

**Cross-references:** {ref}`sec-diagnostics-stability-checks` (Sieve Diagnostic Nodes), Section 24.7 (Scalar Field Diagnostics).

:::{div} feynman-prose
And that's supervised topology. We've seen how class labels become geometry, how classification becomes physics, and how the whole thing fits together with the rest of the framework.

The key insight, the thing I want you to take away, is this: **classification is not about drawing decision boundaries. It's about organizing the internal representation so that similar things are near and different things are far.** The decision boundary is a consequence of this organization, not the primary object.

When you think about it that way, a lot of things make sense. Why does representation learning help classification? Because it organizes the space. Why do contrastive losses work? Because they enforce the "different things should be far" part. Why do we care about the metric? Because distance is how we measure "near" and "far."

The math in this section---the potentials, the jump rates, the losses---is just making these intuitions precise and trainable. But the core idea is simple: classification is geometry.
:::
