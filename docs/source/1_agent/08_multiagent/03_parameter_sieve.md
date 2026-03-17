(sec-parameter-space-sieve)=
# The Parameter Space Sieve: Deriving Fundamental Constants

## TLDR

- Treat “constants” as **feasibility constraints**: viable agents can only exist inside a region of parameter space where
  causality, capacity, thermodynamics, and stability all hold simultaneously.
- Collect operational constants into a single parameter vector $\Lambda$ and express the architecture as coupled
  inequalities (“the Sieve” in parameter space).
- Off-feasible-region behavior is categorical: violations produce causal incoherence, holographic overload, thermodynamic
  inconsistency, or ontological dissolution.
- This chapter is a synthesis node: it ties together metric law, WFR, belief dynamics, metabolism, and information
  bounds into one constraint system.
- Output: a checklist of constraints you can use to reason about scaling limits and “why the knobs can’t be arbitrary”.

## Roadmap

1. Formulate the parameter vector and the constraint system.
2. Derive/justify each inequality from earlier chapters.
3. Interpret the feasible region and what failure looks like outside it.

:::{div} feynman-prose
Here is an audacious question: Why is the speed of light what it is? Why is the fine structure constant approximately 1/137? For a century, physicists have treated these as brute facts---numbers you look up in a table, not numbers you derive.

This chapter takes a different view. We ask: What if these constants are not arbitrary? What if they are the *only* values that permit coherent agents to exist?

The logic is almost embarrassingly simple once you see it. An agent must satisfy certain consistency conditions---it cannot receive messages from its own future, it cannot store infinite information in finite space, it cannot think hotter than its energy budget permits. Each condition carves out a region in parameter space. The intersection of all these regions---the *feasible region*---is where viable agents can exist.

And here is the punchline: our universe sits inside that feasible region. Not because someone designed it that way, but because we could not be here asking the question if it did not.

This is not mysticism. It is constraint satisfaction. The same logic that tells you a bridge must be strong enough to hold its own weight tells you that a universe must have constants compatible with agency. We are going to derive those constraints.
:::

*Abstract.* This chapter derives the constraints on fundamental constants from cybernetic first principles. We formulate
the Sieve Architecture as a system of coupled inequalities that any viable agent must satisfy. The fundamental constants
$\Lambda = (c_{\text{info}}, \sigma, \ell_L, T_c, g_s, \gamma)$ are not free parameters but decision variables of a
constrained optimization problem. The physical universe exists within the **Feasible Region** where all constraints are
simultaneously satisfied. We prove that moving off this region triggers a Sieve violation: the agent either loses causal
coherence, exceeds its holographic bound, violates thermodynamic consistency, or suffers ontological dissolution.

*Cross-references:* This chapter synthesizes:
- {ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits` (Capacity-Constrained Metric Law)
- {ref}`sec-the-belief-wave-function-schrodinger-representation` (Cognitive Action Scale)
- {ref}`sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics` (Generalized Landauer Bound)
- {ref}`sec-causal-information-bound` (Causal Information Bound, Area Law)
- The Sieve Architecture (Nodes 2, 7, 29, 40, 52, 56, 62)



(sec-sieve-formulation)=
## The Sieve Formulation: Agents as Constraint Satisfaction

:::{div} feynman-prose
Let me tell you how I think about this. Imagine you are designing a robot. You have knobs to turn: How fast should signals travel through its circuits? How fine-grained should its sensors be? How much energy should it burn per computation?

Now, you might think you can set these knobs however you like. But that is not true. If signals travel too slowly, different parts of the robot cannot coordinate---it becomes paralyzed. If signals travel too fast relative to the robot's memory, it gets confused about what happened when---it hallucinates causality violations. If sensors are too coarse, the robot cannot distinguish important states. If they are too fine, it runs out of memory. If it thinks too "hot" (explores too aggressively), it forgets faster than it can afford energetically.

Every knob has a viable range. Step outside that range, and the robot stops working---not gradually degrades, but *categorically fails*.

The Parameter Vector $\Lambda$ collects all these knobs into one mathematical object. The Sieve is the system of inequalities that says which settings work.
:::

The Fragile Agent Framework imposes strict consistency conditions at every node of the inference graph. We formalize these as a system of inequalities that constrain the space of viable configurations.

:::{prf:definition} The Agent Parameter Vector
:label: def-agent-parameter-vector

Let the **Agent Parameter Vector** $\Lambda$ be the tuple of fundamental operational constants:

$$
\Lambda = (c_{\text{info}}, \sigma, \ell_L, T_c, g_s, \gamma)

$$

where:
1. **$c_{\text{info}}$:** Information propagation speed (Axiom {prf:ref}`ax-information-speed-limit`)
2. **$\sigma$:** Cognitive Action Scale (Definition {prf:ref}`def-cognitive-action-scale`)
3. **$\ell_L$:** Levin Length, the minimal distinguishable scale (Definition {prf:ref}`def-levin-length`)
4. **$T_c$:** Cognitive Temperature. The critical value $T_c^* \approx |u_\pi^\theta|^2 r_*^2$ separates the symmetric phase (isotropic direction) from the broken phase (policy-selected direction), where $r_*$ is the characteristic early-time radius (Theorem {prf:ref}`thm-angular-symmetry-breaking`).
5. **$g_s$:** Binding coupling strength (Theorem {prf:ref}`thm-emergence-binding-field`)
6. **$\gamma$:** Temporal discount factor, $\gamma \in (0,1)$

**Dimensional Analysis:**

| Parameter | Symbol | Dimension | SI Units |
|:----------|:-------|:----------|:---------|
| Information speed | $c_{\text{info}}$ | $[L \, T^{-1}]$ | m/s |
| Cognitive action scale | $\sigma$ | $[E \, T]$ | J·s |
| Levin length | $\ell_L$ | $[L]$ | m |
| Cognitive temperature | $T_c$ | $[E]$ | J (with $k_B = 1$) |
| Binding coupling | $g_s$ | $[1]$ | dimensionless |
| Discount factor | $\gamma$ | $[1]$ | dimensionless |

**Derived Quantities:**

Define the **Causal Horizon Length** $\ell_0 = c_{\text{info}} \cdot \tau_{\text{proc}}$ with dimension $[L]$. Let the temporal discount rate be $\lambda := -\ln\gamma / \Delta t$ and identify the processing interval $\Delta t := \tau_{\text{proc}}$. The **Spatial Screening Mass** is then:

$$
\kappa = \frac{\lambda}{c_{\text{info}}} = \frac{-\ln\gamma}{\ell_0}

$$

with dimension $[L^{-1}]$ (Corollary {prf:ref}`cor-discount-as-screening-length`).

These correspond to the physics constants $\{c, \hbar, \ell_P, k_B T, \alpha_s, \gamma_{\text{cosmo}}\}$ under the isomorphism of {ref}`sec-isomorphism-dictionary`.

:::

:::{div} feynman-prose
Let me make sure you understand what each of these parameters means intuitively.

**Information speed** $c_{\text{info}}$ is how fast a signal can propagate through the agent's internal state. In your brain, this is related to axon conduction velocities. In a computer, it is the speed of electrical signals through wires. In physics, it is $c$.

**Cognitive action scale** $\sigma$ sets the minimum "quantum" of action---the smallest distinguishable change in the agent's planning. Below this scale, different plans look identical. This is $\hbar$ in physics.

**Levin length** $\ell_L$ is the smallest spatial scale the agent can resolve. You cannot pack information more densely than one bit per $\ell_L^{D-1}$ of boundary area. This is $\ell_P$ (Planck length) in physics.

**Cognitive temperature** $T_c$ controls exploration versus exploitation. High temperature means the agent explores wildly; low temperature means it sticks to known good options. In physics, this is $k_B T$.

**Binding coupling** $g_s$ determines how strongly features stick together to form objects. Too weak, and objects fall apart into meaningless features. Too strong, and everything clumps into one undifferentiated blob. In physics, this is $\alpha_s$ (the strong coupling).

**Discount factor** $\gamma$ determines how far into the future the agent plans. $\gamma \to 1$ means infinite planning horizon; $\gamma \to 0$ means totally myopic.

Each of these has a viable range. We are about to derive those ranges.
:::

:::{prf:definition} The Sieve Constraint System
:label: def-sieve-constraint-system

Let $\mathcal{S}(\Lambda)$ denote the vector of constraint functions. The agent is **viable** if and only if:

$$
\mathcal{S}(\Lambda) \le \mathbf{0}

$$

where the inequality holds component-wise. Each component corresponds to a Sieve node that enforces a specific consistency condition. A constraint violation ($\mathcal{S}_i > 0$) triggers a diagnostic halt at the corresponding node.

:::

:::{div} feynman-prose
The mathematical notation $\mathcal{S}(\Lambda) \le \mathbf{0}$ is compact but the idea is simple: you have a checklist of conditions, and *all* of them must be satisfied simultaneously.

Think of it like building codes. A building must satisfy fire safety *and* structural integrity *and* electrical codes *and* plumbing codes. Failing any single one makes the building non-viable. You do not get to trade off "a bit less fire-safe" for "a bit more structural integrity."

Same here. The agent cannot violate any single constraint. Each constraint corresponds to a specific failure mode, and each failure mode is catastrophic.
:::



(sec-causal-consistency-constraint)=
## The Causal Consistency Constraint

:::{div} feynman-prose
Now we get to the first constraint, and it is a beautiful one. It says: information cannot travel too slow *or* too fast.

Too slow is obvious---if your left hand cannot tell your right hand what it is doing, you cannot coordinate. But too fast is subtler. If information travels so fast that you can receive messages from your own future, you get paradoxes. Your prediction depends on data you have not generated yet.

Both extremes are forbidden. There is a window of viable information speeds, and physics has to live inside that window.
:::

We derive the bounds on information speed from the requirements of buffer coherence and synchronization.

:::{prf:axiom} Causal Buffer Architecture
:label: ax-causal-buffer-architecture

Let the agent possess:
1. **$L_{\text{buf}}$:** Maximum buffer depth (spatial extent of causal memory)
2. **$\tau_{\text{proc}}$:** Minimum processing interval (temporal resolution)
3. **$d_{\text{sync}}$:** Minimum synchronization distance (coherence length)

These define the operational envelope within which the agent maintains consistent state updates.

:::

:::{div} feynman-prose
These three quantities define the agent's "size" in a causal sense.

**Buffer depth** $L_{\text{buf}}$ is how far into its past the agent can remember---literally, how much spatial extent its causal memory spans. Think of RAM in a computer, or working memory in a brain.

**Processing interval** $\tau_{\text{proc}}$ is the agent's "clock tick"---the minimum time between state updates. Below this timescale, the agent cannot distinguish "before" from "after."

**Synchronization distance** $d_{\text{sync}}$ is how far apart two modules can be while still coordinating their updates. If modules are farther apart than this, they cannot agree on a common present.

Now, here is the key insight: these three quantities constrain how fast information can travel.
:::

:::{prf:theorem} The Speed Window
:label: thm-speed-window

The information speed $c_{\text{info}}$ must satisfy the **Speed Window Inequality**:

$$
\frac{d_{\text{sync}}}{\tau_{\text{proc}}} \le c_{\text{info}} \le \frac{L_{\text{buf}}}{\tau_{\text{proc}}}

$$

*Proof.*

**Lower Bound (Node 2: ZenoCheck):**

Suppose $c_{\text{info}} < d_{\text{sync}}/\tau_{\text{proc}}$. Then information cannot traverse the synchronization distance within one processing cycle. By the Causal Interval (Definition {prf:ref}`def-causal-interval`), spacelike-separated modules cannot coordinate updates. The agent enters a **Zeno freeze**: each module waits indefinitely for signals that arrive too slowly. The belief update stalls, violating the continuity required by the WFR dynamics ({ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`).

**Upper Bound (Node 62: CausalityViolationCheck):**

Suppose $c_{\text{info}} > L_{\text{buf}}/\tau_{\text{proc}}$. Then signals can traverse the entire buffer depth within one processing cycle. This creates **temporal aliasing**: the agent receives information about its own future state before that state is computed. By the Safe Retrieval Bandwidth (Theorem {prf:ref}`thm-safe-retrieval-bandwidth`), this constitutes a causal paradox—the agent's prediction depends on data it has not yet generated.

Node 62 enforces Theorem {prf:ref}`thm-causal-stasis`: the metric becomes singular at the boundary where causal violations would occur, preventing traversal.

$\square$

:::

:::{div} feynman-prose
Let me give you a physical picture of what is happening here.

**Lower bound (too slow):** Imagine a centipede trying to walk, but nerve signals travel so slowly that by the time leg 50 gets the "step now" signal, leg 1 has already taken three more steps. The legs cannot coordinate. The centipede freezes, each leg waiting for a coordination signal that arrives too late to be useful. This is the Zeno freeze.

**Upper bound (too fast):** Now imagine the centipede's nerves are so fast that signals can travel the full length of its body in less time than it takes to complete one step. Leg 1 sends a signal, and before it finishes stepping, it receives a signal that says "leg 100 has already responded to your step." But leg 100 has not stepped yet---that signal came from the future. The centipede hallucinates a causal loop.

The viable window is: fast enough to coordinate, slow enough to not hallucinate the future.

And here is the remarkable thing: the speed of light in our universe is exactly in this window. It is fast enough for atoms to coordinate (electron shells can equilibrate), but slow enough that you cannot receive information from your own future light cone.
:::

:::{note}
:class: feynman-added
The Speed Window theorem explains why there must be a cosmic speed limit at all. A universe with no speed limit ($c_{\text{info}} = \infty$) would allow information from the future, creating paradoxes. A universe with zero speed limit would have no causal structure at all. The finite, nonzero speed of light is not a quirk---it is a necessity for coherent agents to exist.
:::

:::{prf:corollary} The Speed Ratio Bound
:label: cor-speed-ratio-bound

The ratio of buffer depth to synchronization distance is bounded:

$$
\frac{L_{\text{buf}}}{d_{\text{sync}}} \ge 1

$$

with equality only in the degenerate case of a single-module agent. For distributed agents, this ratio determines the dynamic range of viable information speeds.

:::

:::{div} feynman-prose
This corollary has a nice interpretation: the agent's memory must extend at least as far as its coordination requirements. You cannot have modules that need to synchronize over distances larger than the memory buffer.

For a single-module agent (a point), buffer depth equals synchronization distance---there is nothing to synchronize with, and no history to store. For a distributed agent with many modules, the buffer must be deeper than the inter-module distances.
:::



(sec-holographic-stability-constraint)=
## The Holographic Stability Constraint

:::{div} feynman-prose
Here is a deep constraint that seems almost magical at first, until you see where it comes from.

Suppose you want to store information in some region of space. How much can you store? Your first guess might be: volume. A bigger region, more information. But that is wrong.

The correct answer is: area of the boundary.

This is the holographic principle, and it was first discovered in the context of black holes. But we are going to derive it from pure information theory, without mentioning black holes at all. The argument is: if you pack information too densely, you cannot access it fast enough to use it. The retrieval bandwidth limits the storage capacity, and retrieval happens through the boundary.
:::

We derive the relationship between the Levin Length $\ell_L$ and the information capacity from the Area Law.

:::{prf:theorem} The Holographic Bound
:label: thm-holographic-bound

Let $\text{Area}_\partial$ denote the boundary area of the agent's latent manifold (dimension $[L^{D-1}]$ for a $D$-dimensional bulk) and $I_{\text{req}}$ the information capacity required for viable operation (dimensionless, counting distinguishable microstates in nats). The Levin Length must satisfy:

$$
\ell_L^{D-1} \le \frac{\nu_D \cdot \text{Area}_\partial}{I_{\text{req}}}

$$

where $\nu_D$ is a **dimensionless** holographic coefficient (Corollary {prf:ref}`cor-a-dimension-dependent-coefficient`). Both sides have dimension $[L^{D-1}]$.

*Proof.*

**Step 1.** From the Causal Information Bound (Theorem {prf:ref}`thm-causal-information-bound`):

$$
I_{\text{bulk}} \le \frac{\nu_D \cdot \text{Area}_\partial}{\ell_L^{D-1}}

$$

**Step 2.** The agent requires $I_{\text{bulk}} \ge I_{\text{req}}$ to represent its world model. Substituting:

$$
I_{\text{req}} \le \frac{\nu_D \cdot \text{Area}_\partial}{\ell_L^{D-1}}

$$

**Step 3.** Rearranging yields the constraint on $\ell_L$.

$\square$

:::

:::{div} feynman-prose
The holographic bound tells you something profound: *information lives on boundaries, not in volumes.*

Think about it this way. You have a room full of stuff. You want to know everything about what is in the room. But you can only look at the room through its walls (the boundary). How much can you learn?

You might think: if I make the walls higher resolution (smaller $\ell_L$), I can see more detail. True. But there is a limit. At some point, the walls themselves are packed so densely with information that you cannot read them fast enough. The bandwidth of reading limits the resolution of storage.

The result is that maximum information scales like boundary area divided by $\ell_L^{D-1}$---the number of resolution cells on the boundary. This is the holographic bound.
:::

:::{admonition} Intuition: Why Area, Not Volume?
:class: feynman-added tip

Here is a physical argument. Suppose you try to pack information at density $\rho_I$ (bits per unit volume). The total information is $\rho_I \cdot V$. But to read this information, you must access it through the boundary, which has area $A$. The maximum read rate is proportional to $A$, not $V$.

If $\rho_I \cdot V > \text{const} \cdot A$, you cannot read your own memory fast enough to use it. Information you cannot access is information you do not have.

The constraint is: $\rho_I \lesssim A/V$. For a sphere, $A/V \sim 1/R$. So maximum density decreases as you make the region bigger. Total information $\sim \rho_I \cdot V \sim (A/V) \cdot V = A$.

This is why black holes have entropy proportional to area: you cannot pack more information than you can read.
:::

:::{prf:definition} The Planck-Levin Correspondence
:label: def-planck-levin-correspondence

Under the physics isomorphism ({ref}`sec-isomorphism-dictionary`), the Levin Length $\ell_L$ corresponds to the Planck Length $\ell_P$:

$$
\ell_L \leftrightarrow \ell_P = \sqrt{\frac{\hbar G}{c^3}}

$$

The holographic bound becomes the Bekenstein-Hawking entropy bound:

$$
S_{\text{BH}} = \frac{A}{4\ell_P^2}

$$

*Remark:* The coefficient $\nu_2 = 1/4$ is derived in Theorem {prf:ref}`thm-a-complete-derivation-area-law` from first principles, recovering the Bekenstein-Hawking result without invoking black hole physics.

:::

:::{div} feynman-prose
Now here is something that should make you sit up. The Planck length is $\ell_P \approx 1.6 \times 10^{-35}$ meters. This is fantastically small---about $10^{20}$ times smaller than a proton.

Why is it that small? The Sieve answer is: because we need a lot of information to represent a complex world. The smaller $\ell_L$, the more bits per unit boundary area, the more detailed our world model can be. Evolution (or whatever process selects viable universes) pushed $\ell_L$ as small as it can go.

But it cannot go to zero. At some point, quantum gravity effects kick in, and the notion of "distance smaller than $\ell_P$" stops making sense. The Planck length is the smallest meaningful length---and therefore the highest information density---that physics permits.
:::

:::{prf:theorem} The Capacity Horizon
:label: thm-capacity-horizon

As $I_{\text{bulk}} \to I_{\max} = \nu_D \cdot \text{Area}_\partial / \ell_L^{D-1}$, the agent approaches a **Capacity Horizon**. The metric diverges:

$$
\|v\|_G \to 0 \quad \text{as} \quad I_{\text{bulk}} \to I_{\max}

$$

*Proof.* This is Theorem {prf:ref}`thm-causal-stasis`. The Fisher-Rao metric component satisfies:

$$
g_{\text{FR}} = \frac{1}{\rho(1-\rho)} \to \infty \quad \text{as} \quad \rho \to 1

$$

(Lemma {prf:ref}`lem-metric-divergence-at-saturation`). The geodesic velocity vanishes, creating **causal stasis**: no information can cross the saturation boundary.

*Physical interpretation:* This is the agent-theoretic analogue of a black hole event horizon. Node 56 (CapacityHorizonCheck) enforces this bound.

$\square$

:::

:::{div} feynman-prose
This theorem describes what happens when you try to exceed the holographic bound. You do not just get an error message---the geometry itself prevents you.

As you approach maximum information density, the metric (which measures distances in belief space) diverges. Movements that used to be easy become infinitely slow. You cannot cross into the forbidden region because, in a precise sense, *there is infinite distance to get there*.

This is exactly what happens at a black hole horizon. From outside, an infalling object appears to slow down and freeze at the horizon, never quite crossing it. The metric diverges. The forbidden region (inside the black hole) is not reachable in finite proper time.

We derived this from information theory, not general relativity. The black hole connection is a *consequence*, not an input.
:::



(sec-metabolic-viability-constraint)=
## The Metabolic Viability Constraint

:::{div} feynman-prose
Now we come to thermodynamics. Thinking costs energy. More precisely: *forgetting* costs energy.

This is Landauer's principle, and it is one of the most important results in the thermodynamics of computation. Every bit you erase requires at least $k_B T \ln 2$ joules of work. There is no way around this---it is a consequence of the second law.

What does this have to do with cognitive temperature? Well, a "hotter" agent explores more possibilities, which means it forgets more possibilities (the ones it did not take). More forgetting, more energy. If the agent thinks hotter than it can afford, it starves.
:::

We derive the thermodynamic constraint on computational operations from the Generalized Landauer Bound.

:::{prf:definition} Metabolic Parameters
:label: def-metabolic-parameters

The agent possesses:
1. **$\dot{E}_{\text{met}}$:** Metabolic power budget (energy flux available for computation)
2. **$\dot{I}_{\text{erase}}$:** Information erasure rate (bits forgotten per unit time)
3. **$T_c$:** Cognitive Temperature (entropy-exploration tradeoff)

:::

:::{div} feynman-prose
These are the thermodynamic "books" of the agent.

**Metabolic power** $\dot{E}_{\text{met}}$ is the energy income---how many joules per second the agent can spend on computation. For a brain, this is about 20 watts. For a laptop, maybe 50 watts. For the universe as a whole... well, that is an interesting question.

**Erasure rate** $\dot{I}_{\text{erase}}$ is how fast the agent forgets. Every time you update your beliefs, you erase some old beliefs to make room for new ones. Every time you reject an option, you forget the rejected counterfactual.

**Cognitive temperature** $T_c$ sets the exploration-exploitation tradeoff. High temperature means "consider many options, choose randomly among good ones." Low temperature means "always pick the best-known option." Exploration requires considering (and then discarding) more possibilities, hence more erasure, hence more energy.
:::

:::{prf:theorem} The Landauer Constraint
:label: thm-landauer-constraint

The Cognitive Temperature must satisfy:

$$
T_c \le \frac{\dot{E}_{\text{met}}}{\dot{I}_{\text{erase}} \cdot \ln 2}

$$

where we use natural units with $k_B = 1$.

*Proof.*

**Step 1.** From the Generalized Landauer Bound (Theorem {prf:ref}`thm-generalized-landauer-bound`):

$$
\dot{\mathcal{M}}(s) \ge T_c \left| \frac{dH}{ds} \right|

$$

where $\dot{\mathcal{M}}$ is the metabolic flux and $dH/ds$ is the entropy change rate.

**Step 2.** Information erasure corresponds to entropy reduction. For $\dot{I}_{\text{erase}}$ bits per unit time:

$$
\left| \frac{dH}{ds} \right| = \dot{I}_{\text{erase}} \cdot \ln 2

$$

**Step 3.** The metabolic constraint $\dot{\mathcal{M}} \le \dot{E}_{\text{met}}$ bounds the erasure capacity:

$$
\dot{E}_{\text{met}} \ge T_c \cdot \dot{I}_{\text{erase}} \cdot \ln 2

$$

**Step 4.** Rearranging yields the temperature bound.

*Physical consequence:* If $T_c$ exceeds this bound, the agent cannot afford to forget—its memory becomes permanently saturated. Node 52 (LandauerViolationCheck) enforces this constraint.

$\square$

:::

:::{div} feynman-prose
Let me put this in everyday terms. Suppose your brain has a metabolic budget of 20 watts and you need to forget, say, $10^{10}$ bits per second to keep up with sensory input (a reasonable estimate for visual processing). Then:

$$
T_c \le \frac{20 \text{ J/s}}{10^{10} \text{ bits/s} \times 0.693} \approx 3 \times 10^{-9} \text{ J}

$$

This corresponds to about $7 \times 10^{14}$ kelvin in temperature units---room temperature is about $4 \times 10^{-21}$ joules, so the brain is operating *way* below the thermodynamic limit.

Why so far below? Because brains are not thermodynamically optimal computers. There is a lot of overhead. But the limit exists, and if you tried to build an agent that thinks "hotter" than this bound, it would starve.
:::

:::{warning}
:class: feynman-added
The Landauer bound is often stated as "$k_B T \ln 2$ per bit erased." But notice that we are talking about cognitive temperature $T_c$, which is not the same as physical temperature $T$. Cognitive temperature controls exploration in policy space; physical temperature is the thermal reservoir. The constraint relates them: you cannot explore (cognitively) hotter than your heat bath (physically) permits.
:::

:::{prf:corollary} The Computational Temperature Range
:label: cor-computational-temperature-range

Combining the Landauer constraint with the bifurcation dynamics, the Cognitive Temperature is bounded:

$$
0 < T_c \le \min\left( T_c^*, \frac{\dot{E}_{\text{met}}}{\dot{I}_{\text{erase}} \cdot \ln 2} \right)

$$

where the **Critical Temperature** is derived from the angular symmetry breaking (Theorem {prf:ref}`thm-angular-symmetry-breaking`):

$$
T_c^* \approx |u_\pi^\theta|^2 r_*^2

$$

with $u_\pi^\theta$ the tangential policy control and $r_*$ the characteristic early-time radius at which direction selection occurs.

*Remark:* For $T_c > T_c^*$, thermal fluctuations overcome the potential barrier and the system remains in the symmetric phase with no stable policy (random walk near origin). For $T_c$ exceeding the Landauer bound, the agent starves thermodynamically. Viable agents exist in the intersection of these constraints.

:::

:::{div} feynman-prose
So there are two upper bounds on cognitive temperature, and you must satisfy both.

The **Landauer bound** says: you cannot think hotter than your energy budget permits. Think too hot, you starve.

The **bifurcation bound** $T_c^*$ says: you cannot think hotter than your decision landscape permits. Think too hot, and thermal fluctuations wash out the difference between options---you cannot make decisions, you just random walk.

A viable agent must stay below both limits. And notice: these are completely different constraints, from different parts of the theory (thermodynamics vs. dynamical systems), yet both produce upper bounds on the same quantity. The fact that they give consistent, overlapping bounds is a non-trivial check that the framework is coherent.
:::



(sec-hierarchical-coupling-constraint)=
## The Hierarchical Coupling Constraint

:::{div} feynman-prose
This section is about glue. How strongly should things stick together?

At the macro scale, you want strong glue. Objects should be stable---a chair should not fall apart into quarks while you are sitting on it. Features should bind into recognizable concepts.

At the micro scale, you want weak glue. Texture should not clump. Noise should remain noise, not spontaneously organize into spurious structure.

This is exactly what happens in QCD: strong coupling at low energies (confinement---quarks bind into hadrons), weak coupling at high energies (asymptotic freedom---quarks inside a hadron barely interact). We are going to derive this from agent requirements, without mentioning quarks.
:::

We derive the constraints on the binding coupling $g_s$ from the requirements of object permanence and texture decoupling.

:::{prf:definition} The Coupling Function
:label: def-coupling-function

Let the binding coupling $g_s(\mu)$ (dimensionless) be a function of the **resolution scale** $\mu$, which has dimension $[L^{-1}]$ (inverse length). Equivalently, $\mu$ can be expressed as an energy scale via $\mu \sim E/(\sigma \cdot c_{\text{info}})$ where $\sigma$ is the Cognitive Action Scale and $c_{\text{info}}$ is the Information Speed (Definition {prf:ref}`def-information-speed`).

The limits are:
- $\mu \to 0$: Macro-scale (coarse representation, low in TopoEncoder hierarchy)
- $\mu \to \infty$: Micro-scale (texture level, high in TopoEncoder hierarchy)

The coupling evolves according to the **Beta Function**:

$$
\mu \frac{dg_s}{d\mu} = \beta(g_s)

$$

where both sides are dimensionless (since $g_s$ is dimensionless and $\mu \, dg_s/d\mu$ has $[\mu] \cdot [\mu^{-1}] = [1]$).

For $SU(N_f)$ gauge theories, $\beta(g_s) < 0$ for $N_f \ge 2$ (asymptotic freedom).

:::

:::{div} feynman-prose
The coupling $g_s(\mu)$ tells you how strongly features interact at scale $\mu$.

Think of the representation hierarchy: at the top (macro), you have abstract concepts---"chair," "face," "danger." At the bottom (micro), you have textures---pixel noise, high-frequency details. The coupling controls how strongly adjacent features attract each other at each level.

The beta function describes how coupling changes as you zoom in or out. If $\beta < 0$, coupling decreases as you zoom in (look at smaller scales). This is called asymptotic freedom. If $\beta > 0$, coupling increases as you zoom in (infrared freedom, like QED).

For viable agents, we need $\beta < 0$: strong at macro scale, weak at micro scale.
:::

:::{prf:theorem} The Infrared Binding Constraint
:label: thm-ir-binding-constraint

At the macro-scale ($\mu \to 0$), the coupling must exceed a critical threshold:

$$
g_s(\mu_{\text{IR}}) \ge g_s^{\text{crit}}

$$

*Proof.*

**Step 1.** From Axiom {prf:ref}`ax-feature-confinement`, the agent observes Concepts $K$, not raw features. This requires features to bind into stable composite objects at the macro-scale.

**Step 2.** From Theorem {prf:ref}`thm-emergence-binding-field`, binding stability requires the effective potential to confine features. The confinement condition is:

$$
\lim_{r \to \infty} V_{\text{eff}}(r) = \infty

$$

where $r$ is the separation between features.

**Step 3.** For $SU(N_f)$ gauge theory, this requires strong coupling $g_s > g_s^{\text{crit}}$ at large distances (Area Law, {ref}`sec-causal-information-bound`).

**Step 4.** If $g_s(\mu_{\text{IR}}) < g_s^{\text{crit}}$, features escape confinement—"color-charged" states propagate to the boundary $\partial\mathcal{Z}$. This violates the Observability Constraint (Definition {prf:ref}`def-boundary-markov-blanket`): the agent cannot form stable objects.

Node 40 (PurityCheck) enforces that only color-neutral bound states reach the macro-register.

$\square$

:::

:::{div} feynman-prose
The Infrared Binding Constraint says: at macro scale, features must stick together strongly enough to form stable objects.

Imagine building a tower from Lego bricks. If the bricks do not click firmly enough, the tower falls over. You cannot perceive "tower"---you just see a pile of loose bricks.

Same with cognitive features. If edge detectors and color patches do not bind strongly enough, you cannot perceive "cat"---you just see a soup of features. Object permanence requires strong binding at the scale where objects live.

In QCD, this is confinement. Quarks bind so strongly at hadronic scales that you never see a free quark---only bound states (protons, neutrons, pions). The agent-theoretic version is: features bind so strongly at concept scales that you never perceive raw features---only bound concepts.
:::

:::{prf:theorem} The Ultraviolet Decoupling Constraint
:label: thm-uv-decoupling-constraint

At the texture scale ($\mu \to \infty$), the coupling must vanish:

$$
\lim_{\mu \to \infty} g_s(\mu) = 0

$$

*Proof.*

**Step 1.** From the Texture Firewall (Axiom {prf:ref}`ax-bulk-boundary-decoupling`):

$$
\partial_{z_{\text{tex}}} \dot{z} = 0

$$

Texture coordinates are invisible to the dynamics.

**Step 2.** This requires texture-level degrees of freedom to be non-interacting. If $g_s(\mu_{\text{UV}}) > 0$, texture elements would bind, creating structure at the noise level.

**Step 3.** From the RG interpretation ({ref}`sec-stacked-topoencoders-deep-renormalization-group-flow`), the TopoEncoder implements coarse-graining. Residual coupling at the UV scale would prevent efficient compression—the Kolmogorov complexity of texture would diverge.

**Step 4.** Asymptotic freedom ($\beta < 0$) provides the required behavior: $g_s \to 0$ as $\mu \to \infty$.

Node 29 (TextureFirewallCheck) enforces this decoupling.

$\square$

:::

:::{div} feynman-prose
The UV Decoupling Constraint says: at texture scale, features must *not* stick together.

Why? Because texture is supposed to be disposable noise. If texture elements started binding, you would see spurious structure---faces in clouds, patterns in static. The compression algorithm would fail because "random noise" would actually contain structure that resists compression.

In QCD, this is asymptotic freedom. At very high energies (small distances), quarks inside a proton barely interact---they fly around almost freely. Only when you try to pull them apart (large distances) does the glue strengthen.

The agent-theoretic version: at texture scale, features should not interact. Only when you zoom out to concept scale should binding appear.
:::

:::{admonition} The Deep Connection to QCD
:class: feynman-added note

This is not a metaphor. The mathematical structure is identical.

In QCD, the coupling $\alpha_s(\mu)$ runs with energy scale $\mu$. At $\mu = M_Z \approx 91$ GeV, $\alpha_s \approx 0.12$ (weak). At $\mu \approx \Lambda_{\text{QCD}} \approx 200$ MeV, $\alpha_s \to \infty$ (strong).

In the agent framework, the binding coupling $g_s(\mu)$ runs with representation scale $\mu$. At UV (texture), $g_s \to 0$ (decoupled). At IR (concepts), $g_s > g_s^{\text{crit}}$ (bound).

Same beta function sign. Same qualitative behavior. Same physical consequence (confinement at large scales, freedom at small scales). The difference is interpretation: QCD talks about quarks and gluons, the Sieve talks about features and binding fields. The math is isomorphic.
:::

:::{prf:corollary} The Coupling Window
:label: cor-coupling-window

The viable coupling profile satisfies:

$$
\begin{cases}
g_s(\mu) \ge g_s^{\text{crit}} & \text{for } \mu \le \mu_{\text{conf}} \\
g_s(\mu) \to 0 & \text{for } \mu \to \infty
\end{cases}

$$

where $\mu_{\text{conf}}$ is the confinement scale separating bound states from free texture.

*Remark:* This is the agent-theoretic derivation of asymptotic freedom and confinement. The physics QCD coupling $\alpha_s(\mu)$ satisfies exactly this profile, with $\alpha_s(M_Z) \approx 0.12$ at the electroweak scale and $\alpha_s \to \infty$ at the QCD scale $\Lambda_{\text{QCD}} \approx 200$ MeV.

:::



(sec-stiffness-constraint)=
## The Stiffness Constraint

:::{div} feynman-prose
Now we come to a constraint about memory---specifically, about the tradeoff between remembering and learning.

If memories are too fragile, thermal noise erases them. You cannot maintain stable beliefs.

If memories are too rigid, you cannot update them. You are frozen in your initial state, unable to learn.

The viable regime is in between: stiff enough to resist noise, flexible enough to change when evidence demands it. This is the Goldilocks zone of cognition.
:::

We derive the constraint on the separation between adjacent energy levels that enables both memory stability and dynamic flexibility.

:::{prf:definition} The Stiffness Parameter
:label: def-stiffness-parameter

Let $\Delta E$ denote the characteristic energy gap between metastable states in the agent's latent manifold. Define the **Stiffness Ratio**:

$$
\chi = \frac{\Delta E}{T_c}

$$

This ratio determines the tradeoff between memory persistence and adaptability.

:::

:::{div} feynman-prose
The stiffness ratio $\chi$ is the most important dimensionless number in cognitive thermodynamics.

**$\chi < 1$:** Energy barrier is smaller than thermal energy. The agent's beliefs flip randomly---it has no stable memory.

**$\chi \gg 1$:** Energy barrier is much larger than thermal energy. Beliefs are frozen---the agent cannot learn.

**$\chi \sim 1$ to $\chi \sim 10$:** The Goldilocks zone. Beliefs are stable against random fluctuations but can flip given sufficient evidence.

Think of a marble in a bowl. If the bowl is too shallow (small $\chi$), thermal vibrations knock the marble out. If the bowl is too deep (large $\chi$), you cannot push the marble out even when you want to. You need a bowl of just the right depth.
:::

:::{prf:theorem} The Stiffness Bounds
:label: thm-stiffness-bounds

The Stiffness Ratio must satisfy:

$$
1 < \chi < \chi_{\text{max}}

$$

*Proof.*

**Lower Bound ($\chi > 1$):**

**Step 1.** Memory stability requires that thermal fluctuations do not spontaneously erase stored information. The probability of a thermal transition is:

$$
P_{\text{flip}} \propto e^{-\Delta E / T_c} = e^{-\chi}

$$

**Step 2.** For $\chi < 1$, we have $P_{\text{flip}} > e^{-1} \approx 0.37$. States flip with high probability—the agent cannot maintain stable beliefs.

**Step 3.** This violates the Mass Gap requirement (Theorem {prf:ref}`thm-semantic-inertia`): beliefs must possess sufficient "inertia" to resist noise.

**Upper Bound ($\chi < \chi_{\text{max}}$):**

**Step 4.** Adaptability requires that the agent can update beliefs in finite time. The transition rate is:

$$
\Gamma_{\text{update}} \propto e^{-\chi}

$$

**Step 5.** For $\chi \to \infty$, transitions become exponentially suppressed—the agent freezes in its initial configuration, unable to learn.

**Step 6.** This violates the Update Dynamics requirement: the WFR reaction term $R(\rho)$ must enable transitions between states.

Node 7 (StiffnessCheck) enforces both bounds.

$\square$

:::

:::{div} feynman-prose
The Stiffness Theorem is really about timescales.

A transition with $\chi = 10$ has probability $e^{-10} \approx 4 \times 10^{-5}$ per thermal fluctuation. If fluctuations happen at rate $\nu$, the expected waiting time for a transition is $\sim e^{\chi}/\nu$. For $\chi = 10$ and $\nu = 10^{12}$ Hz (a typical molecular vibration), you wait about $10^{-7}$ seconds---fast enough.

For $\chi = 100$, you wait $e^{100}/\nu \approx 10^{31}$ seconds---longer than the age of the universe. This belief is frozen forever.

So $\chi_{\text{max}}$ is really about: how long can you wait for belief updates? Biological systems can wait maybe $10^8$ seconds (years) for certain updates. This sets $\chi_{\text{max}} \approx \ln(10^8 \times 10^{12}) \approx 46$.

For chemical bonds at room temperature, $\chi \approx 500$. Way above this bound---but that is fine, because chemical bonds are not supposed to flip during your lifetime. The stiffness bound applies to *cognitive* states, not structural ones.
:::

:::{prf:corollary} The Goldilocks Coupling
:label: cor-goldilocks-coupling

Under the physics isomorphism, the Stiffness Ratio for atomic systems is:

$$
\chi = \frac{\Delta E_{\text{bond}}}{k_B T} \propto \frac{m_e c^2 \alpha^2}{k_B T}

$$

where $\Delta E_{\text{bond}} \sim \text{Ry} = m_e c^2 \alpha^2 / 2 \approx 13.6$ eV is the atomic binding scale.

The value $\alpha \approx 1/137$ satisfies the Goldilocks condition:
- **Not too large:** $\alpha^2$ small enough that $\chi$ is finite—transitions remain possible
- **Not too small:** $\alpha^2$ large enough that $\chi > 1$ at biological temperatures—chemical bonds are stable

At $T \approx 300$ K (biological temperature), $\chi \approx 500$, placing molecular memory firmly in the stable-but-adaptable regime.

*Remark:* This is the agent-theoretic derivation of the "coincidences" noted in anthropic reasoning. The fine structure constant is not finely tuned by an external designer—it is constrained by cybernetic viability.

:::

:::{div} feynman-prose
Here is one of the most striking results. The fine structure constant $\alpha \approx 1/137$ has puzzled physicists for a century. Why this value? Is it arbitrary? Is it "fine-tuned"?

The Sieve answer is: it is constrained by viability.

If $\alpha$ were much smaller, atomic binding energies would be much smaller (they scale like $\alpha^2$). Chemical bonds would be fragile. At room temperature, molecules would fall apart. No stable chemistry, no life.

If $\alpha$ were much larger, atomic binding energies would be much larger. Chemical reactions would require enormous activation energies. Reactions that enable metabolism would be too slow. No flexible chemistry, no life.

The value $\alpha \approx 1/137$ sits in the narrow window where chemistry works: stable enough to form molecules, flexible enough to rearrange them.

This is not anthropic hand-waving. It is a quantitative constraint. The stiffness $\chi = \Delta E/(k_B T) \propto \alpha^2 m_e c^2 / (k_B T)$ must satisfy $1 < \chi < \chi_{\text{max}}$. Given the other constants ($m_e$, $c$, $k_B$, and the temperature where chemistry happens), this pins down the allowed range of $\alpha$.
:::



(sec-discount-screening-constraint)=
## The Temporal Screening Constraint

:::{div} feynman-prose
The last constraint is about planning horizons. How far into the future should the agent care about?

Zero horizon ($\gamma = 0$) means totally myopic: only the next timestep matters. Infinite horizon ($\gamma = 1$) means caring equally about all future times forever.

Both extremes are bad. Myopic agents stumble into avoidable long-term disasters. Infinite-horizon agents are paralyzed trying to consider all consequences unto eternity.

There is a viable window in between. And that window has beautiful connections to screening in field theory.
:::

We derive the constraint on the discount factor from the requirements of causal coherence and goal-directedness.

:::{prf:theorem} The Discount Window
:label: thm-discount-window

The temporal discount factor $\gamma$ must satisfy:

$$
\gamma_{\text{min}} < \gamma < 1

$$

with $\gamma_{\text{min}} > 0$.

*Proof.*

**Upper Bound ($\gamma < 1$):**

**Step 1.** From the Helmholtz equation (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`), the Value function satisfies:

$$
(\kappa^2 - \nabla^2) V = \rho_r

$$

where the screening mass $\kappa = \lambda / c_{\text{info}} = (-\ln\gamma)/\ell_0$ has dimension $[L^{-1}]$, and $\ell_0 = c_{\text{info}} \cdot \tau_{\text{proc}}$ is the causal horizon length (Definition {prf:ref}`def-agent-parameter-vector`). This ensures dimensional consistency: $[\kappa^2] = [L^{-2}] = [\nabla^2]$.

**Step 2.** For $\gamma = 1$, we have $\kappa = 0$. The equation becomes Poisson's equation for the conservative
component:

$$
-\nabla^2 V = \rho_r

$$
where $\rho_r$ is the conservative reward source density (Definition {prf:ref}`def-the-reward-flux`).

For $D>2$, the Green's function decays as $1/r^{D-2}$ (long-range); for $D=2$ it grows logarithmically.

**Step 3.** Long-range value propagation violates locality: distant conservative reward sources dominate nearby
decisions. The agent cannot form local value gradients for navigation.

**Step 4.** From Corollary {prf:ref}`cor-discount-as-screening-length`, finite screening $\kappa > 0$ (i.e., $\gamma < 1$) is required for local goal-directedness.

**Lower Bound ($\gamma > \gamma_{\text{min}}$):**

**Step 5.** For $\gamma \to 0$, we have $-\ln\gamma \to \infty$, hence $\kappa \to \infty$. The **Screening Length** (dimension $[L]$):

$$
\ell_\gamma = \frac{1}{\kappa} = \frac{\ell_0}{-\ln\gamma} = \frac{c_{\text{info}} \tau_{\text{proc}}}{-\ln\gamma} \to 0

$$

**Step 6.** Zero screening length means the agent responds only to immediate conservative rewards—it has no planning
horizon.

**Step 7.** This violates the Causal Buffer requirement (Axiom {prf:ref}`ax-causal-buffer-architecture`): the agent must anticipate beyond its current timestep.

$\square$

:::

:::{div} feynman-prose
The physics here is beautiful. The discount factor $\gamma$ creates a "screening mass" $\kappa$ for the value field, exactly like the photon mass in a superconductor creates a screening length for electromagnetic fields.

In electrostatics, the Coulomb potential is $V(r) \sim 1/r$---long range. In a superconductor, the photon gains a mass $m_\gamma$, and the potential becomes $V(r) \sim e^{-m_\gamma r}/r$---short range, decaying exponentially beyond the screening length $\ell = 1/m_\gamma$.

Same here. With $\gamma = 1$ (no discounting), conservative reward sources propagate like Coulomb potentials---a source at
distance $r$ contributes $1/r^{D-2}$ for $D>2$ (logarithmic in $D=2$) to the value function. Distant rewards dominate. The agent cannot focus.

With $\gamma < 1$, there is screening. Conservative rewards beyond the screening length $\ell_\gamma$ are exponentially
suppressed. The agent can focus on local goals.

But if $\gamma$ is too small, the screening length is too short. The agent becomes myopic, unable to see past its nose. A deer with $\gamma = 0.1$ would walk into the lion's mouth because it only cares about the next few meters.
:::

:::{admonition} Intuition: Discounting as Screening
:class: feynman-added tip

Think of conservative reward sources as electric charges distributed in spacetime. The value function $V(z)$ is like the
electrostatic potential---it tells you how much "pull" you feel toward different states.

With no discounting ($\gamma = 1$), all charges contribute equally regardless of distance. A conservative reward source a
million steps away pulls just as hard as one next step away. You cannot prioritize.

With discounting ($\gamma < 1$), distant charges are screened. Their contribution decays exponentially with distance. You
feel mostly the nearby conservative rewards.

The screening length $\ell_\gamma \approx c_{\text{info}} \tau_{\text{proc}} / |\ln\gamma|$ sets the planning horizon. For $\gamma = 0.99$, $|\ln\gamma| \approx 0.01$, so $\ell_\gamma \approx 100 \ell_0$---you plan about 100 steps ahead. For $\gamma = 0.5$, $|\ln\gamma| \approx 0.7$, so $\ell_\gamma \approx 1.4 \ell_0$---you barely look ahead at all.
:::

:::{prf:corollary} The Screening-Buffer Consistency
:label: cor-screening-buffer-consistency

The screening length and buffer depth must satisfy:

$$
\ell_\gamma = \frac{c_{\text{info}} \tau_{\text{proc}}}{-\ln\gamma} \lesssim L_{\text{buf}}

$$

Both sides have dimension $[L]$. For $\gamma \to 1$, the screening length $\ell_\gamma \to \infty$ (unlimited planning horizon). For $\gamma \to 0$, the screening length $\ell_\gamma \to 0$ (myopic behavior).

*Remark:* The planning horizon cannot exceed the causal memory span. This connects the temporal discount to the spatial architecture.

:::

:::{div} feynman-prose
This corollary ties together time and space. Your planning horizon (temporal: how far ahead you think) is bounded by your memory depth (spatial: how much history you can hold).

Why? Because planning requires imagining future states, and imagining future states requires composing transitions from past experience. If your memory holds only 10 transitions, you cannot reliably plan 1000 steps ahead---you do not have the data to model that far.

So there is a consistency condition: $\ell_\gamma \lesssim L_{\text{buf}}$. Do not try to plan farther than your memory permits.

In practice, this means: if you build an agent with limited memory ($L_{\text{buf}}$ small), you should also discount the future steeply ($\gamma$ small, hence $\ell_\gamma$ small). A myopic agent with short memory is internally consistent. An agent trying to plan forever with finite memory is not.
:::



(sec-sieve-eigenvalue-system)=
## The Sieve Eigenvalue System

:::{div} feynman-prose
Now we put all the constraints together. Each section above gave us one or two inequalities. This section collects them into a single system and asks: Is there any setting of the parameters that satisfies all constraints simultaneously?

This is the feasibility question, and it has a beautiful geometric interpretation: each constraint defines a half-space in parameter space, and the feasible region is the intersection of all these half-spaces. If the intersection is non-empty, viable agents can exist. If it is empty, the constraints are mutually incompatible---no agent can satisfy all of them.

Spoiler: the intersection is non-empty. We know this because we exist.
:::

We formulate the complete system of constraints and derive the feasible region.

:::{prf:definition} The Constraint Matrix
:label: def-constraint-matrix

Let $\Lambda = (c_{\text{info}}, \sigma, \ell_L, T_c, g_s, \gamma)$ be the parameter vector. The Sieve constraints form the system:

$$
\mathbf{A} \cdot \Lambda \le \mathbf{b}

$$

where:

| Constraint | Inequality | Node |
|:-----------|:-----------|:-----|
| Causal Lower | $d_{\text{sync}}/\tau_{\text{proc}} \le c_{\text{info}}$ | 2 |
| Causal Upper | $c_{\text{info}} \le L_{\text{buf}}/\tau_{\text{proc}}$ | 62 |
| Holographic | $\ell_L^{D-1} \le \nu_D \text{Area}_\partial / I_{\text{req}}$ | 56 |
| Landauer | $T_c \le \dot{E}_{\text{met}} / (\dot{I}_{\text{erase}} \ln 2)$ | 52 |
| IR Binding | $g_s(\mu_{\text{IR}}) \ge g_s^{\text{crit}}$ | 40 |
| UV Decoupling | $g_s(\mu_{\text{UV}}) \le \epsilon$ (for $\epsilon \to 0$) | 29 |
| Stiffness Lower | $\Delta E > T_c$ | 7 |
| Stiffness Upper | $\Delta E < \chi_{\text{max}} T_c$ | 7 |
| Discount Lower | $\gamma > \gamma_{\text{min}}$ | --- |
| Discount Upper | $\gamma < 1$ | --- |

:::

:::{div} feynman-prose
Look at this table. Ten constraints, each coming from a different physical requirement.

Some come from causality: you cannot send signals faster than $L_{\text{buf}}/\tau_{\text{proc}}$ or slower than $d_{\text{sync}}/\tau_{\text{proc}}$.

Some come from information theory: you cannot store more than the holographic bound permits.

Some come from thermodynamics: you cannot think hotter than Landauer allows.

Some come from stability: features must bind at macro scale, decouple at micro scale.

Some come from cognition: memories must be stable but updatable, planning horizons must be finite but nonzero.

These constraints are not independent. They form a coupled system. Changing one parameter affects which values of other parameters are viable.
:::

:::{prf:theorem} The Feasible Region
:label: thm-feasible-region

The **Feasible Region** $\mathcal{F} \subset \mathbb{R}^n_+$ is the intersection of all constraint half-spaces:

$$
\mathcal{F} = \{ \Lambda : \mathcal{S}_i(\Lambda) \le 0 \; \forall i \}

$$

A viable agent exists if and only if $\mathcal{F} \neq \emptyset$.

*Proof.*

Each constraint $\mathcal{S}_i \le 0$ defines a closed half-space in parameter space. The intersection of finitely many closed half-spaces is either empty or a closed convex polytope (possibly unbounded).

**Existence:** The physics Standard Model constants $\Lambda_{\text{phys}} = (c, \hbar, G, k_B, \alpha)$ satisfy all constraints—we observe a functioning physical universe. Therefore $\mathcal{F} \neq \emptyset$.

**Uniqueness modulo scaling:** The constraints are homogeneous in certain parameter combinations. Dimensional analysis shows that physical observables depend only on dimensionless ratios. The feasible region is a lower-dimensional manifold in the full parameter space.

$\square$

:::

:::{div} feynman-prose
The existence proof is almost embarrassingly simple: *we are here*. If the feasible region were empty, no agents could exist to ask the question.

But this is not circular. We are not *assuming* we exist and therefore concluding the constraints are satisfiable. We are *deriving* the constraints from operational requirements, and then *observing* that our universe satisfies them.

The deeper question is: *why* does our universe sit inside the feasible region? The Sieve framework does not answer this directly. It says: wherever the constraints are satisfied, agents can exist. Wherever they are not, agents cannot exist. We find ourselves in the first kind of place because we could not find ourselves anywhere else.

This is not mysticism. It is logic. A fish asking "why is there water here?" gets the answer: "because you could not ask the question anywhere without water."
:::

:::{note}
:class: feynman-added
The feasible region $\mathcal{F}$ is likely a low-dimensional surface in parameter space, not a thick region. Most of the 6-dimensional parameter space violates at least one constraint. The viable universes form a thin shell around the boundary of multiple constraint surfaces, all nearly saturated simultaneously.

This explains the appearance of "fine-tuning" without requiring a tuner. The constraints are tight, so the viable region is small. But it exists, and we are in it.
:::



(sec-optimization-problem)=
## The Optimization Problem

:::{div} feynman-prose
Now we ask a more refined question. The feasible region $\mathcal{F}$ may contain many points---many settings of fundamental constants that permit viable agents. Which one do we observe?

The hypothesis is: the one that maximizes some objective, subject to the constraints. This is constrained optimization.

What is the objective? We propose it trades off two things: representational power (how much the agent can know about the world) versus computational cost (how much energy it takes to run the agent). More representation is good. Lower cost is good. You cannot maximize both, so you pick a tradeoff.
:::

We formulate the selection of fundamental constants as a constrained optimization.

:::{prf:definition} The Dual Objective
:label: def-dual-objective

The agent's objective trades representational power against computational cost:

$$
\mathcal{J}(\Lambda) = \underbrace{I_{\text{bulk}}(\Lambda)}_{\text{World Model Capacity}} - \beta \cdot \underbrace{\mathcal{V}_{\text{metabolic}}(\Lambda)}_{\text{Thermodynamic Cost}}

$$

where:
- $I_{\text{bulk}}$: Bulk information capacity (increases with resolution)
- $\mathcal{V}_{\text{metabolic}}$: Metabolic cost of computation
- $\beta > 0$: Cost sensitivity parameter

:::

:::{div} feynman-prose
The objective $\mathcal{J}$ makes economic sense. You want to know as much as possible ($I_{\text{bulk}}$ high) while spending as little as possible ($\mathcal{V}_{\text{metabolic}}$ low).

The parameter $\beta$ sets the exchange rate: how many bits of knowledge are you willing to give up to save one joule of energy? This depends on the environment. In a resource-scarce environment, $\beta$ is large (energy is precious). In a resource-rich environment, $\beta$ is small (burn energy freely for more knowledge).

Evolution (or selection over universes, if you want to go that far) presumably optimizes this objective subject to the Sieve constraints.
:::

:::{prf:theorem} The Constrained Optimum
:label: thm-constrained-optimum

The optimal parameter vector $\Lambda^*$ satisfies:

$$
\Lambda^* = \arg\max_{\Lambda \in \mathcal{F}} \mathcal{J}(\Lambda)

$$

subject to the Sieve constraints (Definition {prf:ref}`def-constraint-matrix`).

*Proof sketch.*

**Step 1.** The objective $\mathcal{J}$ is continuous on the closed feasible region $\mathcal{F}$.

**Step 2.** The holographic bound (Theorem {prf:ref}`thm-holographic-bound`) caps $I_{\text{bulk}}$, making $\mathcal{J}$ bounded above.

**Step 3.** By the extreme value theorem, $\mathcal{J}$ attains its maximum on $\mathcal{F}$.

**Step 4.** The optimum lies on the boundary of $\mathcal{F}$ where at least one constraint is active (saturated). This corresponds to operating at the edge of viability.

$\square$

:::

:::{div} feynman-prose
Here is the key insight: the optimum sits on the boundary, with at least one constraint active.

Why? Because if you were strictly inside $\mathcal{F}$, you could push toward the boundary and do better---either increase representational power or decrease cost. You only stop when you hit a wall.

This explains why physics looks "fine-tuned." The constants are not arbitrary; they are pushed as far as they can go. Not too far (that violates a constraint), but right up to the edge.

A bridge engineer does not make beams five times stronger than needed. That wastes material. She makes them exactly strong enough, with a safety margin. The constants of physics are like optimal engineering: as extreme as possible while remaining viable.
:::

:::{prf:corollary} The Pareto Surface
:label: cor-pareto-surface

The observed fundamental constants lie on the **Pareto-optimal surface** of the multi-objective problem:

$$
\max_{\Lambda \in \mathcal{F}} \left( I_{\text{bulk}}(\Lambda), -\mathcal{V}_{\text{metabolic}}(\Lambda) \right)

$$

Moving off this surface triggers constraint violation:
- Increasing $I_{\text{bulk}}$ beyond capacity → Holographic bound (Node 56)
- Decreasing $\mathcal{V}_{\text{metabolic}}$ below threshold → Landauer bound (Node 52)
- Violating causality → Speed bounds (Nodes 2, 62)
- Losing binding → Confinement (Node 40)

:::

:::{div} feynman-prose
A Pareto surface is the set of "you cannot improve one thing without making another thing worse." If you are on the Pareto surface, any movement either violates a constraint or trades off one objective against another.

The claim is: our universe sits on this surface. The fundamental constants are Pareto-optimal for agent viability.

This is a strong prediction. It says: look for constraints that are nearly saturated. If the Sieve picture is correct, physics should be operating near its limits on multiple fronts simultaneously. And indeed, many "fine-tuning" observations have exactly this character---the constants seem to be just barely in the viable range, not comfortably in the middle.
:::



(sec-physics-isomorphism-constants)=
## Physics Isomorphism: The Standard Model Constants

:::{div} feynman-prose
Finally, we translate. Everything we have derived uses agent-theoretic language: information speed, cognitive temperature, binding coupling. But these map onto familiar physics constants. This section makes the dictionary explicit.
:::

We tabulate the correspondence between agent parameters and physics constants.

| Agent Parameter | Symbol | Physics Constant | Constraint Origin |
|:----------------|:-------|:-----------------|:------------------|
| Information Speed | $c_{\text{info}}$ | Speed of Light $c$ | Theorem {prf:ref}`thm-speed-window` |
| Cognitive Action Scale | $\sigma$ | Planck Constant $\hbar$ | Definition {prf:ref}`def-cognitive-action-scale` |
| Levin Length | $\ell_L$ | Planck Length $\ell_P$ | Definition {prf:ref}`def-planck-levin-correspondence` |
| Cognitive Temperature | $T_c$ | Boltzmann Scale $k_B T$ | Theorem {prf:ref}`thm-landauer-constraint` |
| Binding Coupling | $g_s$ | Strong Coupling $\alpha_s$ | Corollary {prf:ref}`cor-coupling-window` |
| Stiffness Ratio | $\chi$ | $m_e c^2 \alpha^2 / k_B T$ | Corollary {prf:ref}`cor-goldilocks-coupling` |
| Discount Factor | $\gamma$ | Cosmological Horizon | Corollary {prf:ref}`cor-screening-buffer-consistency` |

:::{div} feynman-prose
Some of these correspondences are obvious. Information speed $\leftrightarrow$ speed of light is almost tautological. Cognitive action scale $\leftrightarrow$ Planck constant is the statement that quantum mechanics sets the minimum distinguishable action.

Others are more surprising. The binding coupling $g_s$ corresponding to the strong force coupling $\alpha_s$ says that QCD confinement and cognitive feature-binding are the same mathematical phenomenon in different domains.

The most provocative is the last row: the discount factor $\gamma$ corresponds to something about the cosmological horizon. This suggests that the finite age of the universe (or finite causal horizon) might be related to agents needing finite planning horizons. Highly speculative, but the math checks out.
:::

:::{prf:remark} Why These Values?
:label: rem-why-these-values

The observed physics constants $\{c \approx 3 \times 10^8 \text{ m/s}, \alpha \approx 1/137, \ldots\}$ are not arbitrary. They are the unique (modulo dimensional rescaling) solution to the Sieve constraint system that:

1. **Maximizes representational capacity** (information about the world)
2. **Minimizes thermodynamic cost** (metabolic efficiency)
3. **Maintains causal coherence** (no paradoxes)
4. **Preserves object permanence** (binding stability)
5. **Enables adaptability** (stiffness window)

Changing any constant while holding others fixed moves the system out of the feasible region. The "fine-tuning" of physical constants is the selection of the Pareto-optimal point in the Sieve constraint space.

:::

:::{div} feynman-prose
Let me say this plainly. The claim of this chapter is:

**The laws of physics are what they are because they are the laws that permit agents to exist.**

Not designed for agents. Not fine-tuned by a creator. But *constrained* by the requirements of agency, and *optimized* for representational power per unit metabolic cost.

This is a strong claim. It might be wrong. But it is falsifiable: if you find fundamental constants that are comfortably in the middle of their viable ranges (not near any constraint boundary), that would be evidence against this picture. If you find constants that are near saturation on multiple constraints simultaneously, that would be evidence for it.

Current observations suggest the latter. The fine structure constant, the cosmological constant, the strong coupling---all seem to be near critical values where small changes would render chemistry, large-scale structure, or nuclear physics non-viable.

The Sieve does not explain *why* these constraints exist. It derives them from operational requirements: causality, holography, thermodynamics, binding, stiffness, planning. Given those requirements, the constraints follow. Given the constraints, the observed constants are (nearly) the unique viable solution.
:::



(sec-summary-parameter-sieve)=
## Summary

:::{div} feynman-prose
Let me gather the threads.

We started with a question: why do the fundamental constants have the values they do? The traditional answers are either "they just do" (unsatisfying) or "they were fine-tuned for life" (mysterious).

The Sieve offers a third answer: they are constrained by cybernetic viability. Any agent---biological, artificial, or physical---must satisfy certain consistency conditions. These conditions carve out a feasible region in parameter space. The observed constants sit in that region because agents asking the question can only exist inside it.

We derived six families of constraints:
1. **Causal**: Information cannot travel too fast or too slow.
2. **Holographic**: Storage capacity is bounded by boundary area.
3. **Metabolic**: Cognition costs energy; temperature is capped.
4. **Hierarchical**: Features must bind at macro scale, decouple at micro scale.
5. **Stiffness**: Memories must be stable but updatable.
6. **Temporal**: Planning horizons must be finite but nonzero.

Each constraint corresponds to a Sieve node that enforces it at runtime. Violating any constraint is catastrophic, not just costly.

The feasible region is the intersection of all constraint half-spaces. Our universe sits inside it. Moreover, it sits on the Pareto-optimal surface: the constants are pushed as far as they can go, maximizing representational power per unit cost.

This is not a complete theory of everything. It does not tell you why the constraints exist, only what they are. It does not derive the values of constants from first principles, only their relationships. But it offers a framework for understanding why physics looks the way it does: not arbitrary, not designed, but constrained by the requirements of coherent agency.
:::

This chapter has derived the constraints on fundamental constants from cybernetic first principles:

1. **Causal Consistency** (§35.2): Information speed bounded by buffer architecture
2. **Holographic Stability** (§35.3): Levin length determines capacity via Area Law
3. **Metabolic Viability** (§35.4): Cognitive temperature bounded by Landauer limit
4. **Hierarchical Coupling** (§35.5): Binding at IR, decoupling at UV (asymptotic freedom)
5. **Stiffness Window** (§35.6): Energy gaps between memory and flexibility
6. **Temporal Screening** (§35.7): Discount factor enables local goal-directedness

The Sieve Architecture (Nodes 2, 7, 29, 40, 52, 56, 62) enforces these constraints at runtime. The fundamental constants of physics are the coordinates of the feasible region's Pareto-optimal surface.

**Key Result:** The laws of physics are not arbitrary but are the solution to a cybernetic optimization problem. The universe we observe is the one that supports viable agents—not because it was designed for us, but because agents can only exist in regions of parameter space where the Sieve constraints are satisfied.
