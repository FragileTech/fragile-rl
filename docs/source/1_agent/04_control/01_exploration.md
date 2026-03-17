(sec-intrinsic-motivation-maximum-entropy-exploration)=
# Intrinsic Motivation: Maximum-Entropy Exploration

## TLDR

- Maximum-entropy (MaxEnt) exploration is **not random dithering**: it is a control objective that preserves future
  reachability (“keep options open”).
- In practice this becomes entropy/KL-regularized control on the macro state-action trajectory space: a temperature
  trades reward for diversity.
- The key identity is a **duality**: maximizing path entropy under reward constraints is equivalent to soft optimal
  control with a KL penalty toward a reference policy.
- Use Sieve diagnostics to prevent MaxEnt failure modes (chattering/Zenoness, over-mixing, loss of grounding).
- This chapter sets up the belief-dynamics and coupling-window chapters: exploration pressure must remain within the
  information-stability window.

## Roadmap

1. Path entropy and exploration gradients (what is optimized).
2. Duality with soft optimality (why MaxEnt control is “just” KL control).
3. Practical guidance: temperatures, horizons, and diagnostic failure modes.

:::{div} feynman-prose
Let me tell you about a beautiful idea that connects two things you might not have thought were related: exploring the world and keeping your options open.

When you learn reinforcement learning, exploration often gets treated as a necessary evil. You need to try different things so you don't get stuck in a local optimum, but exploration is just noise you add to your policy, right? Random actions to shake things loose.

Wrong. There's a deeper way to think about exploration, and it leads to much better algorithms.

Here's the key insight: exploration isn't about randomness for its own sake. It's about **maintaining reachability**. A good explorer is an agent that can still get to many different places in the future. An agent that's painted itself into a corner---even if that corner looks pretty good right now---has lost something valuable. It's lost the *ability to change course*.

Maximum-entropy exploration formalizes this. Instead of asking "what action gives me the highest expected reward right now?", we ask "what action gives me the highest expected reward *while preserving my ability to reach many future states*?" The entropy of your future state-action trajectory distribution is a measure of that ability.

And here's the beautiful thing: once you work through the math, maximizing entropy and maximizing soft (KL-regularized) reward turn out to be the *same problem*, viewed from different angles. This is the duality we'll explore in this section.
:::

:::{admonition} Researcher Bridge: Max-Entropy Exploration in Macro Space
:class: info
:name: rb-maxent-exploration
This is the MaxEnt RL idea applied to discrete macro state-action trajectories. Instead of adding a scalar bonus, we maximize the entropy of reachable macro state-action trajectories, which is the discrete version of "keep options open."
:::

:::{div} feynman-prose
The previous layers define representation ($K,z_n,z_{\text{tex}}$), predictive dynamics ($\bar{P}$), and stability/value constraints ($V,G$, Sieve checks). This layer formalizes an **intrinsic exploration pressure** on the discrete macro register: prefer policies that keep the set of reachable future macro state-action trajectories diverse, which supports reachability/controllability and reduces brittle overcommitment to narrow state-action paths.
:::

(sec-path-entropy-and-exploration-gradients)=
## Path Entropy and Exploration Gradients

:::{div} feynman-prose
We're going to work on the **macro model**---the discrete register of high-level states. Why discrete? Because entropy is well-defined for discrete distributions. There's no ambiguity, no reference measures, no gauge choices. The entropy of a distribution over a finite set is just what Shannon said it is.

We start with the macro Markov kernel: the learned transition probabilities $\bar{P}(k' | k, a)$ that tell us how macro-states evolve under actions. This is the causal enclosure we demanded earlier---the macro-symbol alone (plus action) is sufficient to predict the next macro-symbol.
:::

:::{div} feynman-prose
We work on the **macro model** (the discrete register). Assume a macro Markov kernel

$$
\bar{P}(k'\mid k,a),\qquad k,k'\in\mathcal{K},\ a\in\mathcal{A},

$$
which is the learned effective dynamics demanded by Causal Enclosure ({ref}`sec-conditional-independence-and-sufficiency`).
:::

:::{prf:definition} Macro Path Distribution
:label: def-macro-path-distribution

Fix a horizon $H\in\mathbb{N}$ and a (possibly stochastic) policy $\pi(a\mid k)$. The induced distribution over length-$H$ macro state-action trajectories

$$
\xi := (K_t, A_t, K_{t+1}, A_{t+1}, \dots, A_{t+H-1}, K_{t+H})
    \in \mathcal{K}\times(\mathcal{A}\times\mathcal{K})^H

$$
conditioned on $K_t=k$ is

$$
P_\pi(\xi\mid k)
:=
\prod_{h=0}^{H-1}\pi(A_{t+h}\mid K_{t+h})\ \bar{P}(K_{t+h+1}\mid K_{t+h},A_{t+h}).

$$
(For continuous $\mathcal{A}$, interpret $P_\pi(\xi\mid k)$ as a density with respect to the action reference measure.)

:::

:::{div} feynman-prose
What is this definition saying? Fix where you are now (macro-state $k$) and fix a policy (how you choose actions). Then there's a distribution over full $H$-step state-action trajectories: which actions you might take and which macrostates follow. That's $P_\pi(\xi | k)$---the probability of each possible state-action trajectory.

Notice the factorization. The policy term $\pi(a_t\mid k_t)$ selects actions, and the dynamics term $\bar{P}(k_{t+1}\mid k_t,a_t)$ advances the macrostate. Even if the dynamics were deterministic, a stochastic policy would still induce a distribution over state-action paths, and vice versa.
When we define causal path entropy, we only credit randomness from the policy term. Stochasticity in $\bar{P}$ is not under the agent's control, so it does not contribute.
:::

:::{prf:definition} Causal Path Entropy
:label: def-causal-path-entropy

The causal path entropy at $(k,H)$ under $\pi$ is the cumulative policy entropy along paths
$\xi\in\Gamma_H(k)$ induced by $\pi$ and $\bar{P}$:

$$
S_c(k,H;\pi)
:= \sum_{h=0}^{H-1} \mathbb{E}_{\xi\sim P_\pi(\cdot\mid k)}
\left[ H\!\left(\pi(\cdot\mid K_{t+h})\right) \right].

$$
Only policy randomness contributes; stochasticity in $\bar{P}$ does not add entropy credit.
The expectation is taken under the path law induced by $\pi$ and $\bar{P}$.
This quantity is well-typed because the macro register is discrete; for continuous $\mathcal{A}$, interpret
$H(\pi(\cdot\mid k))$ as a differential entropy with respect to the action reference measure.

:::

:::{div} feynman-prose
Now we're getting somewhere. $S_c(k, H; \pi)$ measures how much randomness the agent injects into its future
action choices along the trajectory. High causal entropy means the policy stays spread out at the states it expects
to visit; low entropy means you commit to a narrow action plan.

The key distinction is control. Environmental noise does not count toward $S_c$ because it is not chosen by the
agent. Causal entropy only credits the randomness the agent injects through $\pi$.

I want to emphasize why discreteness matters here. If $\mathcal{A}$ were continuous, we'd have to talk about
differential entropy, which depends on your choice of reference measure. Different reference measures give different
entropies. With discrete actions (and a discrete macro register), there's no such ambiguity. Entropy is entropy.
This is one of the payoffs for the VQ-VAE architecture that quantizes latent states.
:::

:::{prf:definition} Exploration Gradient, metric form
:label: def-exploration-gradient-metric-form

Let $z_{\text{macro}}=e_k\in\mathbb{R}^{d_m}$ denote the code embedding of $k$ ({ref}`sec-the-shutter-as-a-vq-vae`), and let $G$ be the relevant metric on the macro chart ({ref}`sec-second-order-sensitivity-value-defines-a-local-metric`). Define the exploration gradient as the metric gradient of state-action path entropy:

$$
\mathbf{g}_{\text{expl}}(e_k) := T_c\ \nabla_G S_c(k,H;\pi),

$$
where $T_c>0$ is the cognitive temperature ({prf:ref}`def-cognitive-temperature`). Operationally, gradients are taken through the continuous pre-quantization coordinates (straight-through VQ estimator); in the strictly symbolic limit, the gradient becomes a discrete preference ordering induced by $S_c(k,H;\pi)$.

**Interpretation (Exploration / Reachability).** $S_c(k,H;\pi)$ measures how much action-level randomness the
agent injects along trajectories from $k$ under $\pi$. Increasing $S_c$ preserves **agent-controlled reachability**:
the policy avoids committing to a narrow action sequence, independent of environmental stochasticity.

:::

:::{div} feynman-prose
The exploration gradient $\mathbf{g}_{\text{expl}}$ tells you which direction in state space increases your future optionality. It's like a compass pointing toward freedom. The cognitive temperature $T_c$ controls how strongly you weight exploration versus exploitation---high temperature means exploration dominates, low temperature means you mostly follow reward gradients.

Now here's a subtle point. The macro-state $K$ is discrete, but we take gradients through its continuous embedding $e_k$. How does that work? Through the straight-through estimator from the VQ-VAE. In the forward pass, you quantize to discrete codes. In the backward pass, you pretend the quantization was differentiable and flow gradients to the pre-quantization coordinates. It's a hack, but it works beautifully in practice.
:::

(sec-maxent-duality-utility-entropy-regularization)=
## MaxEnt Duality: Utility + Entropy Regularization

:::{div} feynman-prose
We've defined state-action path entropy as a measure of future reachability. Now let's connect this to standard reinforcement learning by showing that maximizing entropy is equivalent to maximizing a certain kind of soft reward.

The setup is familiar: you have an instantaneous reward function $\mathcal{R}(k, a)$ and a discount factor $\gamma$. The twist is that instead of maximizing expected discounted reward, you maximize expected discounted reward *plus* policy entropy. This is the "entropy regularization" or "soft RL" framework.
:::

:::{prf:definition} MaxEnt RL objective on macrostates
:label: def-maxent-rl-objective-on-macrostates

Let $\mathcal{R}(k,a)$ be an instantaneous reward/cost-rate term ({ref}`sec-re-typing-standard-rl-primitives-as-interface-signals`, {ref}`sec-the-hjb-correspondence`) and let $\gamma\in(0,1)$ be the discount factor (dimensionless). The maximum-entropy objective is

$$
J_{T_c}(\pi)
:=
\mathbb{E}_\pi\left[\sum_{t\ge 0}\gamma^t\left(\mathcal{R}(K_t,K^{\text{act}}_t) + T_c\,\mathcal{H}(\pi(\cdot\mid K_t))\right)\right],

$$
where $\mathcal{H}$ is Shannon entropy. This is the standard "utility + entropy regularization" objective.

**Regimes.**
- $T_c\to 0$: $\pi$ collapses toward determinism; behavior can be brittle under distribution shift.
- $T_c\to\infty$: $\pi$ approaches maximal entropy; behavior becomes overly random and may degrade grounding (BarrierScat).
- The useful regime is intermediate: enough entropy to remain robust, enough utility to remain directed.

:::

:::{div} feynman-prose
Let me make sure this is clear. The objective $J_{T_c}(\pi)$ has two terms at each timestep:

1. **Reward**: $\mathcal{R}(K_t, K^{\text{act}}_t)$---how good is the immediate outcome?
2. **Policy entropy**: $T_c \cdot \mathcal{H}(\pi(\cdot | K_t))$---how spread out is your action distribution?

The cognitive temperature $T_c$ trades off these two concerns. When $T_c$ is large, you care a lot about keeping your options open (high entropy policy). When $T_c$ is small, you care mostly about reward and your policy becomes more deterministic.

The extreme cases are instructive:
- $T_c \to 0$: You become a pure reward maximizer. Your policy converges to the greedy action at each state. This can be brittle---if your environment shifts, you have no backup plans.
- $T_c \to \infty$: You become uniformly random. Every action is equally likely. This is maximally robust but you get no actual work done.

The sweet spot is somewhere in between, and finding it is part of the art of RL algorithm design.
:::

:::{prf:proposition} Soft Bellman form, discrete actions
:label: prop-soft-bellman-form-discrete-actions

Assume finite $\mathcal{A}$. Define the soft state value

$$
V^*(k) := \max_{\pi} \ \mathbb{E}\Big[\sum_{t\ge 0}\gamma^t(\mathcal{R}+T_c\mathcal{H})\ \Big|\ K_0=k\Big].

$$
Then $V^*$ satisfies the entropic Bellman fixed point

$$
V^*(k)
=
T_c \log \sum_{a\in\mathcal{A}}
\exp\!\left(\frac{1}{T_c}\left(\mathcal{R}(k,a)+\gamma\,\mathbb{E}_{k'\sim\bar{P}(\cdot\mid k,a)}[V^*(k')]\right)\right),

$$
and the corresponding optimal policy is the softmax policy

$$
\pi^*(a\mid k)\propto
\exp\!\left(\frac{1}{T_c}\left(\mathcal{R}(k,a)+\gamma\,\mathbb{E}[V^*(k')]\right)\right).

$$
*Proof sketch.* Standard convex duality / log-sum-exp variational identity: maximizing expected reward plus entropy yields a softmax (exponential-family) distribution; substituting back produces the log-partition recursion. (This is the "soft"/MaxEnt Bellman equation used in SAC-like methods.)

**Consequence.** The same mathematics can be read as:
1) maximize reward while retaining policy entropy (MaxEnt RL), or
2) maximize reachability/diversity of future macro state-action trajectories (intrinsic motivation).

:::

:::{div} feynman-prose
The soft Bellman equation is gorgeous. Let me walk you through it.

In regular dynamic programming, the Bellman equation says: the value of a state is the maximum over actions of (immediate reward + discounted future value). You pick the best action and that determines your value.

In *soft* dynamic programming, you replace the maximum with a "soft maximum"---the log-sum-exp. Instead of picking one action, you average over all actions, weighted by their exponentiated values. This is the same as maximizing expected value plus entropy.

The log-sum-exp function has a beautiful property: it's a smooth approximation to the maximum. When $T_c$ is small, $T_c \log \sum_a \exp(Q_a / T_c)$ approaches $\max_a Q_a$. When $T_c$ is large, it approaches the average plus some entropy term. The temperature interpolates between "be greedy" and "be uncertain."

The optimal policy falls out automatically: it's the softmax over Q-values (which are immediate reward plus discounted future value). High-value actions get more probability, but every action gets *some* probability, weighted by temperature.
:::

:::{admonition} The Two Faces of MaxEnt
:class: feynman-added tip

The same mathematics has two interpretations that are useful in different contexts:

1. **MaxEnt RL view:** "I want high reward, but I also want to hedge my bets by keeping my policy from being too deterministic."

2. **Intrinsic motivation view:** "I want to stay in regions of state space where many futures are reachable, because that gives me flexibility to adapt."

These are the same objective, just explained differently. The first emphasizes the reward-seeking behavior with entropy as a regularizer. The second emphasizes the exploration/reachability behavior with reward as a guide.

For building intuition: if you're optimizing for a known reward function, think MaxEnt RL. If you're trying to build an agent that can adapt to changing goals, think intrinsic motivation.
:::


(sec-duality-of-exploration-and-soft-optimality)=
## Duality of Exploration and Soft Optimality

(rb-soft-rl-duality)=
:::{admonition} Researcher Bridge: Soft RL Equals Exploration Duality
:class: info
If you know SAC or KL control, this section formalizes why maximizing entropy and optimizing soft value are the same problem. The exploration gradient is just the covariant form of that duality.
:::

:::{div} feynman-prose
Now we're going to make the duality between exploration and soft optimality precise. This is beautiful mathematics, and it has practical consequences for how you think about policy learning.

The claim is strong: maximizing state-action path entropy subject to expected reward constraints is *exactly the same problem* as maximizing expected reward with a KL penalty toward a reference policy. Different objective functions, same optimal solution. This is a convex duality result, and it's the deep reason why entropy regularization "works."
:::

(sec-formal-definitions)=
## Formal Definitions (Path Space, Causal Entropy, Exploration Gradient)

:::{prf:definition} Causal Path Space
:label: def-causal-path-space

For a macrostate $k\in\mathcal{K}$ and horizon $H$, define the future macro state-action path space

$$
\Gamma_H(k)
:=
\left\{(k_0,a_0,k_1,a_1,\dots,a_{H-1},k_H)\in\mathcal{K}^{H+1}\times\mathcal{A}^H : k_0 = k\right\}.

$$
:::

:::{div} feynman-prose
This is just notation. $\Gamma_H(k)$ is the set of all possible $H$-step state-action trajectories starting from $k$. If $\mathcal{A}$ is finite, $\Gamma_H(k)$ has $|\mathcal{A}|^H|\mathcal{K}|^H$ elements.
:::

:::{prf:definition} Path Probability
:label: def-path-probability

$P_\pi(\xi\mid k)$ is the induced state-action path probability from {prf:ref}`def-macro-path-distribution`.

:::

:::{prf:definition} Causal Entropy
:label: def-causal-entropy

$S_c(k,H;\pi)$ is the causal path entropy from {prf:ref}`def-causal-path-entropy`, i.e., the cumulative policy
entropy along the induced path measure $P_\pi(\cdot\mid k)$.

:::

:::{prf:definition} Exploration gradient, covariant form
:label: def-exploration-gradient-covariant-form

On a macro chart with metric $G$ ({ref}`sec-second-order-sensitivity-value-defines-a-local-metric`),

$$
\mathbf{g}_{\text{expl}}(e_k) := T_c\,\nabla_G S_c(k,H;\pi).

$$
:::

:::{div} feynman-prose
Why "covariant form"? Because we're taking gradients with respect to the metric $G$, not with respect to Euclidean coordinates. The metric gradient $\nabla_G$ accounts for the local geometry of state space. This matters when your state space is curved or has non-uniform sensitivity---the natural direction of steepest ascent depends on the metric.
:::

(sec-the-equivalence-theorem)=
## The Equivalence Theorem (Duality of Causal Regulation)

:::{div} feynman-prose
Now for the main event. The following theorem says that three apparently different ways of stating the optimal control problem are actually equivalent. They give the same optimal policy, and their objective values are related by simple transformations.

This is not a "they're approximately the same" result. It's exact equivalence. The same optimal policy arises whether you think about it as MaxEnt control, as KL-regularized state-action trajectory optimization, or as soft Bellman dynamic programming.
:::

:::{prf:theorem} Equivalence of Entropy-Regularized Control Forms; discrete macro
:label: thm-equivalence-of-entropy-regularized-control-forms-discrete-macro

Assume:
1. finite macro alphabet $\mathcal{K}$ and (for simplicity) finite action set $\mathcal{A}$,
2. an enclosure-consistent macro kernel $\bar{P}(k'\mid k,a)$,
3. bounded reward flux $\mathcal{R}(k,a)$,
4. discount factor $\gamma \in (0,1)$ (ensures convergence of infinite-horizon sums).

Then the following are equivalent characterizations of the same optimal control law:

1. **MaxEnt control (utility + freedom):** $\pi^*$ maximizes $J_{T_c}(\pi)$ from {prf:ref}`def-maxent-rl-objective-on-macrostates`.
2. **Exponentially tilted trajectory measure (KL-regularization).** Fix a reference (prior) policy $\pi_0(a\mid k)$ with full support (uniform when $\mathcal{A}$ is finite). Consider the infinite-horizon trajectory measure. The optimal controlled path law admits an exponential-family form relative to the reference measure induced by $\pi_0$ and $\bar{P}$:

   $$
   P^*(\omega\mid K_t=k)\ \propto\
   P_0(\omega \mid k)\,
   \exp\!\left(\frac{1}{T_c}\sum_{h=0}^{\infty}\gamma^h\,\mathcal{R}(K_{t+h},K^{\text{act}}_{t+h})\right),

   $$
   where $P_0(\omega \mid k) := \prod_{h=0}^{\infty}\pi_0(K^{\text{act}}_{t+h}\mid K_{t+h})\,\bar{P}(K_{t+h+1}\mid K_{t+h},K^{\text{act}}_{t+h})$ is the reference trajectory measure, and the normalizer is the (state-dependent) path-space normalizing constant. (For finite-horizon $H$, replace $\infty$ with $H-1$; the equivalence holds for any horizon.)
3. **Soft Bellman optimality:** the optimal value function $V^*$ satisfies the soft Bellman recursion of {prf:ref}`prop-soft-bellman-form-discrete-actions`, and $\pi^*$ is the corresponding softmax policy.

Moreover, the path-space log-normalizer is (up to scaling) the soft value. Gradients of the log-normalizer therefore induce a well-defined exploration direction in any differentiable macro coordinate system. The link between soft optimality and path entropy is cleanest when stated as a KL-regularized variational identity: if $P_0(\omega\mid k)$ denotes the reference trajectory measure induced by $\pi_0$ and $\bar{P}$, then

$$
\log Z(k)
=
\sup_{P(\cdot\mid k)}
\left\{
\frac{1}{T_c}\,\mathbb{E}_{P}\!\left[\sum_{h=0}^{\infty}\gamma^h\,\mathcal{R}\right]
-D_{\mathrm{KL}}(P(\cdot\mid k)\Vert P_0(\cdot\mid k))
\right\},

$$
and the optimizer is exactly the exponentially tilted law {math}`P^*`. In the special case where {math}`P_0` is uniform (or treated as constant), the KL term differs from Shannon path entropy by an additive constant, recovering the standard "maximize entropy subject to expected reward" view. The finite-horizon version replaces $\infty$ with $H-1$; as $H \to \infty$, the finite-horizon solution converges to the stationary infinite-horizon optimum.

*Proof sketch.* Set up the constrained variational problem "maximize path entropy subject to an expected reward constraint." The Euler-Lagrange condition yields an exponential-family distribution on paths. The normalizer obeys dynamic programming and equals the soft value. Differentiating the log-normalizer yields the corresponding exploration-gradient direction.

:::

:::{div} feynman-prose
Let me unpack why this theorem matters.

**Form 1** is how you think about MaxEnt RL day-to-day: maximize reward plus entropy. This is the objective in SAC, Soft Q-Learning, and similar algorithms.

**Form 2** is the path-space view: instead of thinking about policies, think about distributions over entire state-action trajectories. The optimal state-action trajectory distribution is an exponential tilt of the reference distribution, where the tilt factor is the exponential of cumulative reward. High-reward state-action trajectories get exponentially more probability.

**Form 3** is the dynamic programming view: the soft Bellman equation gives you a recursive way to compute optimal values, and the optimal policy is a softmax over Q-values.

The theorem says these are all the same. If you solve one, you've solved them all. The optimal policy $\pi^*$ is identical whether you derive it from Form 1, Form 2, or Form 3.

The key equation is the variational identity at the end: soft value equals the maximum over state-action trajectory distributions of (expected reward minus KL to reference). This is convex optimization, and the KL penalty is what makes the problem tractable. Without regularization, you'd have a hard maximum over state-action trajectories; with KL regularization, you get a smooth log-sum-exp.
:::

:::{admonition} Why the Log-Normalizer Matters
:class: feynman-added note

The log-normalizer $\log Z(k)$ in the variational identity is the soft value function $V^*(k)$ (up to a temperature-dependent scaling). This is a deep fact from statistical mechanics and exponential families.

In statistical mechanics, the log-normalizer of the Boltzmann distribution is the free energy. Here, the log-normalizer of the exponentially tilted trajectory distribution is the soft value. The same mathematical structure appears in both places because both are doing the same thing: trading off "energy" (negative reward) against "entropy" (randomness).

For practical algorithms, this means: if you can efficiently compute or estimate the normalizing constant $Z(k)$, you can extract values and gradients for policy optimization. This is exactly what soft actor-critic does.
:::

::::{admonition} Connection to RL #22: KL-Regularized Policies as Degenerate Exploration Duality
:class: note
:name: conn-rl-22
**The General Law (Fragile Agent):**
MaxEnt control is equivalent to an **Exponentially Tilted Trajectory Measure**:

$$
P^*(\omega|K_t=k) \propto P_0(\omega|k) \exp\!\left(\frac{1}{T_c}\sum_{h=0}^{\infty} \gamma^h \mathcal{R}(K_{t+h}, K^{\text{act}}_{t+h})\right)

$$
The path-space log-normalizer equals the soft value (Theorem {prf:ref}`thm-equivalence-of-entropy-regularized-control-forms-discrete-macro`). This is a **Schrödinger bridge** formulation.

**The Degenerate Limit:**
Use single-step KL penalty instead of path-space tilting. Ignore the trajectory structure.

**The Special Case (Standard RL):**

$$
J(\pi) = \mathbb{E}[R] - \lambda D_{\mathrm{KL}}(\pi \| \pi_0)

$$
This recovers **KL-Regularized Policy Gradient** and exponential family policies.

**What the generalization offers:**
- **Path-space view**: The optimal policy is a Schrödinger bridge between prior and reward-weighted measures
- **Trajectory entropy**: Explores future *macro state-action trajectories* $\omega = (K^{\text{act}}_t, K_{t+1}, K^{\text{act}}_{t+1}, \ldots)$, not just single actions
- **Variational principle**: Soft value = log-partition function of trajectory measure (eq. above)
- **Causal entropy**: $S_c(k, H; \pi)$ measures future reachability under causal interventions
::::

:::{div} feynman-prose
The connection to standard KL-regularized policies is instructive. When you add a KL penalty $D_{\text{KL}}(\pi \| \pi_0)$ to your policy gradient objective, you're doing a single-step approximation to what we're describing here. The full picture is path-space: you're regularizing toward a reference *state-action trajectory* distribution, not just a reference action distribution.

This matters when your MDP has temporal structure. Single-step KL regularization doesn't account for how today's action affects tomorrow's options. Path-space KL regularization does. The Schrödinger bridge formulation makes this crystal clear: you're finding the state-action path distribution closest to your reference that achieves a certain expected reward.

For discrete macro-states, this is computationally tractable because the path space is finite. For continuous states, you'd need approximations---which is why practical algorithms like SAC use the single-step version and rely on temporal-difference learning to propagate future information backward.
:::

## Failure Modes and Diagnostics

MaxEnt exploration is powerful, but it has characteristic failure modes. In the Fragile Agent, these are meant to be
detectable and actionable:

- **Chattering / Zenoness:** entropy pressure can cause rapid action switching. Monitor Zeno-style switching checks and
  add explicit switching penalties or reduce temperature/horizon.
- **Over-mixing (loss of macro identity):** excessive entropy destroys stable macrostates. Monitor mixing/compactness and
  closure diagnostics; enforce the coupling window rather than increasing entropy indefinitely.
- **Premature collapse (no exploration):** temperature too low collapses the policy to a brittle mode. Monitor entropy /
  reachability metrics and reintroduce exploration pressure when coverage shrinks.
- **Ungrounded exploration:** exploring in the internal model without boundary support leads to hallucinated reachability.
  Monitor grounding/closure synchronization; intervene by tightening closure losses or shortening open-loop rollouts.

The intended workflow is: choose $T_c$ (and horizon) as a knob, then let diagnostics decide when that setting is safe for
the current regime.
