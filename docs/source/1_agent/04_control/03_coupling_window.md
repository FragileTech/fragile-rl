(sec-implementation-note-entropy-regularized-optimal-transport-bridge)=
# Implementation Note: Entropy-Regularized Optimal Transport Bridge

## TLDR

- The Schrödinger / entropy-regularized OT bridge is a **path-space view of KL control**: find the closest trajectory
  measure to a reference dynamics subject to boundary beliefs.
- Operationally: it gives you a principled way to connect priors/posteriors (or start/end distributions) without ad-hoc
  interpolation.
- Use it as an implementation tool: it explains why soft policy iteration behaves like a transport problem on
  trajectories.
- This chapter is optional but clarifies the coupling-window theorem by making “regularized control” geometrically
  explicit.

## Roadmap

1. Set up the bridge problem (reference dynamics + endpoint constraints).
2. Show the KL-control equivalence and what it means for implementation.

:::{div} feynman-prose
Before we get into the coupling window theorem, I want to give you a beautiful piece of optional machinery. It's not required for what follows, but if you understand it, you'll see the whole story from a different angle---a path-space angle.

Here's the setup. You have a reference dynamics---say, your learned macro kernel $\bar{P}$, or a diffusion on your latent space. You have two "boundary conditions": a prior belief (where you think you are) and a posterior belief (where the observations say you are). The question is: what's the most natural way to connect these two?

The answer comes from optimal transport theory: find the path measure that's closest (in KL divergence) to your reference dynamics, subject to matching the boundary conditions. This is called a **Schrödinger bridge**, and it's the rigorous version of "most likely flow under entropy regularization."

Why should you care? Because this is exactly what KL-regularized control does, just stated in path space. When you do soft policy iteration, when you use SAC, when you add entropy bonuses to your objective---you're implicitly solving a Schrödinger bridge problem. The path-space view makes this crystal clear.
:::

(rb-kl-control-bridge)=
:::{admonition} Researcher Bridge: KL Control as a Schrödinger Bridge
:class: tip
Entropy-regularized control can be read as an optimal transport problem on trajectories. This is the same math behind soft policy iteration, just framed as a path-measure bridge.
:::

:::{div} feynman-prose
This is optional machinery, but it provides a clean path-space view of KL-regularized control and filtering.
:::

:::{admonition} The Schrödinger Bridge Picture
:class: feynman-added note

**Entropic bridge (Schrödinger bridge) viewpoint.** Given a reference dynamics (e.g. the macro kernel $\bar{P}$, or a continuous diffusion on $\mathcal{Z}_\mu$) and two marginals (a prior belief and a boundary-conditioned posterior), the bridge problem finds the path measure closest in KL to the reference subject to matching the marginals. This is the rigorous "most likely flow under entropy regularization" principle (entropic optimal transport) {cite}`cuturi2013sinkhorn,leonard2014schrodinger`.

In the Fragile Agent:
- the reference process is the world model's internal rollout,
- the boundary observations induce marginal constraints (via the shutter),
- the Sieve imposes feasibility constraints (via projections),

so each training update can be read as an entropic optimal transport step on belief trajectories.
:::



(sec-theorem-the-information-stability-threshold)=
## Theorem: The Information-Stability Threshold (Coupling Window)

:::{div} feynman-prose
Now we arrive at one of the central results: the **coupling window theorem**. This is the formal statement of the Goldilocks principle I mentioned earlier. Your agent must be coupled to its boundary strongly enough to stay grounded in reality, but not so strongly that it loses coherent internal structure.

Let me give you the physical picture first, then we'll state it precisely.

Imagine your agent as a spinning top. The boundary observations are like a hand that occasionally taps the top to keep it aligned. Too few taps, and the top wobbles off into some random orientation---this is **ungrounded inference**. Too many taps, too hard, and the top never settles into a stable spin at all---this is **symbol dispersion**.

The coupling window is the range of tap frequencies and strengths where the top spins stably *and* stays aligned with the external reference. Outside this window, something breaks.

What makes this theorem useful is that the quantities involved are **measurable**. We're not asking you to verify some abstract condition. We're saying: measure the mutual information between observations and macro-states (that's your grounding rate), measure the entropy of your macro distribution (that's your mixing/dispersion), and check that they're in the right relationship. If they're not, a specific diagnostic fires, and you know what went wrong.
:::

(rb-stable-learning-window)=
:::{admonition} Researcher Bridge: The Stable Learning Window
:class: warning
The coupling window is the stability region where representation and dynamics stay grounded. For RL readers, it plays the role of a learning-rate and discount range where updates contract rather than diverge.
:::

:::{div} feynman-prose
The coupling-window view implies a necessary **window condition**: coupling must be strong enough to remain grounded (BoundaryCheck) but not so strong that the macro register loses coherence (dispersion/mixing).

We state this as a rate balance rather than an ill-typed scalar comparison.
:::

:::{prf:definition} Grounding rate
:label: def-grounding-rate

Let $G_t:=I(X_t;K_t)$ be the symbolic mutual information injected through the boundary (Node 13). The *grounding rate* is the average information inflow per step:

$$
\lambda_{\text{in}} := \mathbb{E}[G_t].

$$
Units: $[\lambda_{\text{in}}]=\mathrm{nat/step}$.

:::

:::{div} feynman-prose
What is this $\lambda_{\text{in}}$, really? It's measuring how much your observations tell you about your macro-state. If $\lambda_{\text{in}}$ is high, each observation carries a lot of information about which macro-symbol is active---your boundary is informative. If $\lambda_{\text{in}}$ is low, observations are mostly noise relative to macro-state identity---your boundary is not telling you much.

Think of it as the bandwidth of the "reality channel" into your agent. You need this channel to have enough capacity to correct drift in your internal model.
:::

:::{prf:definition} Mixing rate
:label: def-mixing-rate

Let $S_t:=H(K_t)$ be the macro entropy. The *mixing rate* is the expected entropy growth not attributable to purposeful exploration:

$$
\lambda_{\text{mix}} := \mathbb{E}[(S_{t+1}-S_t)_+].

$$
Units: $[\lambda_{\text{mix}}]=\mathrm{nat/step}$.

:::

:::{div} feynman-prose
And $\lambda_{\text{mix}}$? That's the rate at which your macro-state distribution is spreading out, becoming more uncertain. Some spreading is fine---it's called exploration. But spreading that happens *despite* your observations, spreading that the grounding signal can't counteract, that's bad. It means your internal structure is dissolving.

The $(\cdot)_+$ notation means we only count positive entropy changes. We're interested in how fast the distribution spreads, not how fast it concentrates.
:::

:::{prf:theorem} Information-stability window; operational
:label: thm-information-stability-window-operational

A necessary condition for stable, grounded macrostates is the existence of constants $0<\epsilon<\log|\mathcal{K}|$ such that, along typical trajectories,

$$
\epsilon \le I(X_t;K_t) \quad\text{and}\quad H(K_t)\le \log|\mathcal{K}|-\epsilon,

$$
and the net entropy balance satisfies

$$
\lambda_{\text{in}} \gtrsim \lambda_{\text{mix}}.

$$
Violations correspond to identifiable barrier modes:
- If $I(X;K)\approx 0$: under-coupling - ungrounded inference / decoupling (Mode D.C).
- If $H(K)\approx \log|\mathcal{K}|$: over-coupling or dispersion - symbol dispersion (BarrierScat).

*Remark.* This theorem is intentionally stated at the level of measurable information quantities (Gate Nodes) so it can be audited online; strengthening it to a sufficient condition requires specifying the macro kernel class and a contraction inequality (e.g. log-Sobolev / Doeblin-type conditions).

:::

:::{div} feynman-prose
Let me unpack this theorem in plain language.

**The two inequalities** say:
1. **You must have grounding**: $I(X_t; K_t) \ge \epsilon$ means observations must carry at least $\epsilon$ nats of information about the macro-state. If this fails, you're flying blind---your observations aren't telling you where you are.

2. **You must not have dispersion**: $H(K_t) \le \log|\mathcal{K}| - \epsilon$ means your belief can't be spread uniformly over all macro-states. If this fails, your agent is completely uncertain about everything, which means the macro-symbol has lost all meaning.

**The rate condition** $\lambda_{\text{in}} \gtrsim \lambda_{\text{mix}}$ says the grounding signal must keep up with natural spreading. Information flows in through the boundary; entropy tends to increase through various sources (model uncertainty, stochasticity, numerical errors). If inflow can't keep up with spreading, you'll eventually drift into bad territory.

**The failure modes** are diagnostic gold:
- **Mode D.C (Decoupling)**: Your observations stopped being informative. Maybe your encoder broke. Maybe the environment changed in a way your shutter can't detect. Either way, you're ungrounded.
- **BarrierScat (Dispersion)**: Your belief is spread over everything. Maybe your updates are too aggressive. Maybe there's too much noise. Either way, your symbols have stopped meaning anything.
:::

:::{admonition} Why This Theorem is Useful
:class: feynman-added tip

Notice what's special here: both conditions are **measurable at runtime**. You can compute $I(X_t; K_t)$ from your encoder and decoder. You can compute $H(K_t)$ from your belief distribution. You don't need ground truth, you don't need access to the "real" environment state---everything is defined in terms of quantities the agent can observe and compute.

This is by design. The theorem is meant to be *operational*---something you can actually check, not just a theoretical guarantee that requires omniscience to verify.
:::

::::{admonition} Connection to RL #9: Conservative Q-Learning as Soft Coupling Window
:class: note
:name: conn-rl-9
**The General Law (Fragile Agent):**
The **Coupling Window** (Theorem {prf:ref}`thm-information-stability-window-operational`) imposes a **hard constraint** on information flow:

$$
\epsilon \le I(X_t; K_t) \quad \text{and} \quad H(K_t) \le \log|\mathcal{K}| - \epsilon.

$$
If violated, the Sieve halts execution (BoundaryCheck failure). This ensures that offline data cannot drive the agent into ungrounded regions of state space.

**The Degenerate Limit:**
Replace the hard constraint with a soft Q-value penalty. Allow violations if reward is high enough.

**The Special Case (Standard RL - CQL):**
Conservative Q-Learning {cite}`kumar2020conservative` adds a penalty for out-of-distribution actions:

$$
\min_Q \mathbb{E}_{s \sim \mathcal{D}, a \sim \mu}\left[\log \sum_a \exp Q(s,a)\right] - \mathbb{E}_{s,a \sim \mathcal{D}}[Q(s,a)] + \text{Bellman}.

$$
This softly penalizes overestimation on unseen actions but **does not prevent** the agent from taking them.

**Result:** CQL is the $\epsilon \to 0$ limit where coupling constraints become soft penalties rather than hard firewalls.

**What the generalization offers:**
- **Hard guarantees**: BoundaryCheck halts execution when grounding fails---the agent cannot "pay the fine" and proceed
- **Auditable thresholds**: $I(X_t; K_t)$ is computed at runtime; failures are logged with specific diagnostic codes
- **Information-theoretic grounding**: The constraint is derived from the Data Processing Inequality, not a heuristic penalty term
- **Bidirectional protection**: Both under-coupling (ungrounded) and over-coupling (dispersion) are detected and blocked
::::

:::{div} feynman-prose
The connection to Conservative Q-Learning is illuminating. CQL says: "penalize Q-values for actions you haven't seen data for." That's a soft version of "don't be confident about things you're not grounded in." Our coupling window makes this hard: if you're not grounded, you halt. You can't pay a penalty and proceed into uncharted territory.

Is the hard version better? It depends on your risk tolerance. For safety-critical applications, hard constraints are essential---you don't want your robot to say "I'll take the penalty" and drive off a cliff. For more forgiving domains, soft penalties may give you more flexibility.
:::


(sec-summary-unified-information-theoretic-control-view)=
## Summary: Unified Information-Theoretic Control View

:::{div} feynman-prose
Let me step back and show you the big picture. We've been building up a framework that unifies several different perspectives---geometry, boundaries, exploration, belief dynamics, optimality---under a single information-theoretic roof. Here's how the pieces fit together:
:::

:::{div} feynman-added
| Level | Formalism | Law |
| :--- | :--- | :--- |
| Geometry | Riemannian $(\mathcal{Z},G)$ | Distance measured by a sensitivity metric (Fisher/Hessian) |
| Boundary | Markov blanket $B_t$ ({prf:ref}`def-boundary-markov-blanket`) | Environment = boundary law $P_{\partial}$ ({ref}`sec-definitions-interaction-under-partial-observability`) |
| Exploration | Causal entropy / MaxEnt RL | Reachability pressure via path entropy on $\mathcal{K}$ ({ref}`sec-intrinsic-motivation-maximum-entropy-exploration`) |
| Belief Dynamics | Filtering + projection | Predict - update - project ({ref}`sec-belief-dynamics-prediction-update-projection`) |
| Optimality | Soft Bellman / log-normalizer | Soft value = log-normalizer; exploration gradient from path entropy ({ref}`sec-correspondence-table-filtering-control-template`) |
:::

:::{div} feynman-prose
Each row in this table is a different lens on the same underlying system:

**Geometry** tells you how to measure distances---not in Euclidean coordinates, but in a way that respects the sensitivity structure of your state space. Two points are "far apart" if small perturbations at one don't look like small perturbations at the other.

**Boundary** defines what "the environment" even means. It's not an object with hidden state you can peek at; it's a conditional law relating your actions to your observations. Everything you know about the world passes through this membrane.

**Exploration** is about keeping options open. The path entropy on $\mathcal{K}$ measures how many futures are reachable from where you are. Higher entropy means more freedom, more ability to adapt to unexpected circumstances.

**Belief Dynamics** is the engine that keeps your internal model synchronized with reality. Predict what you expect to see, update based on what you actually see, project away anything that violates your constraints.

**Optimality** ties it all together. The soft Bellman equation says: your value function is a log-normalizer over exponentially weighted paths. Maximizing entropy and maximizing expected reward are dual views of the same optimization problem.

The Fragile Agent is a system that implements all of these layers, with explicit capacity limits and safety constraints woven throughout.
:::

:::{admonition} The Fragile Conclusion
:class: feynman-added important

The agent is a Bounded-Rationality Controller ({prf:ref}`def-bounded-rationality-controller`) with explicit information and stability constraints. Macro symbols remain meaningful only inside the coupling window (Theorem {prf:ref}`thm-information-stability-window-operational`); outside it, the system either exhibits ungrounded inference (under-coupling) or loses macro structure through excessive mixing/dispersion (over-coupling).

This is not a bug to be fixed; it's a feature to be monitored. The boundaries of the coupling window tell you exactly where your agent's competence ends.
:::
