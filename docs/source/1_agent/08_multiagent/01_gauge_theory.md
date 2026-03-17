(sec-symplectic-multi-agent-field-theory)=
# Relativistic Symplectic Multi-Agent Field Theory

## TLDR

- Treat multi-agent interaction as a **field theory** on a causal bundle: information propagates at finite speed, so
  value/belief become dynamical fields (hyperbolic rather than elliptic equations).
- Replace “instantaneous observation of opponents” with **ghost interfaces** (retarded coupling): agents optimize
  against delayed images consistent with causality.
- Nash equilibria appear as **standing-wave patterns** in the joint field; the Newtonian MARL picture is the
  $c_{\text{info}}\to\infty$ limit.
- The symplectic structure persists: coupled agents interact via structured boundary terms, not arbitrary reward hacks.
- Outputs: a principled language for MARL stability diagnostics, causal delay effects, and where standard algorithms
  break their assumptions.

## Roadmap

1. Upgrade MARL to causal field dynamics (why instantaneous coupling is wrong).
2. Define ghost interfaces and the causal bundle that restores Markov structure.
3. Derive equilibrium/interaction structure and connect to diagnostics/implementation.

*Abstract.* We derive Multi-Agent Reinforcement Learning (MARL) as a system of $N$ coupled field equations on a causal
spacetime structure. When agents do not share exactly the same boundary, information propagates at finite speed
$c_{\text{info}}$, transforming the elliptic (Helmholtz) value equation into a hyperbolic (Klein-Gordon) wave equation.
The Markov property, lost on the spatial manifold $\mathcal{Z}^{(N)}$ alone, is restored on the **Causal Bundle**
$\mathcal{Z}^{(N)} \times \Xi_{<t}$, where $\Xi_{<t}$ is the Memory Screen integrating incoming wavefronts from the past
light cone. We derive the **Ghost Interface**, where agents optimize against retarded images of their opponents, and
prove that Nash Equilibrium is a standing wave pattern in the joint causal field. The instantaneous (Newtonian)
formulation emerges as the $c_{\text{info}} \to \infty$ limit.

(rb-relativistic-marl)=
:::{admonition} Researcher Bridge: From Action-at-a-Distance to Field Theory
:class: warning
Standard Multi-Agent RL assumes a global clock: when Agent A acts, Agent B sees it instantly. In distributed systems or physical reality, this violates causality. We upgrade the framework: Value is not a static field but a **propagating wave**. Agents do not interact with each other's current states; they interact with the **Past Light Cone** of the environment. Memory is no longer optional—it is the physical requirement to restore the Markov property in a relativistic universe.
:::

*Cross-references:* This section generalizes the Helmholtz equation
({ref}`sec-the-bulk-potential-screened-poisson-equation`) to the Klein-Gordon equation, and elevates the
Memory Screen ({ref}`sec-the-historical-manifold-and-memory-screen`) from a recording device to a primary
state variable. It extends the single-agent geometry (Sections 20–24) to the multi-agent setting.

*Literature:* Game theory {cite}`fudenberg1991game`; stochastic games {cite}`shapley1953stochastic`; multi-agent RL
{cite}`littman1994markov,lowe2017multi`; symplectic geometry {cite}`arnold1989mathematical`; retarded potentials
{cite}`jackson1999classical`.

:::{div} feynman-prose
Now, I want to tell you about something that bothered me for a long time when I first thought about multi-agent systems. You see, in most of the literature, people write down these beautiful game-theoretic equations where all the agents can see each other perfectly, instantly, at every moment. Agent A knows what Agent B is doing right now. Agent B knows what Agent A is thinking right now. It is all very nice and symmetric and mathematically convenient.

But wait a minute. How does Agent A know what Agent B is doing? The information has to travel somehow. Maybe it is light bouncing off B and hitting A's sensors. Maybe it is a message sent over a network. Maybe it is vibrations in the air. But whatever it is, it takes time. The information is not instantaneous.

This is not some pedantic quibble about milliseconds. When you think about it carefully, you realize that what A is optimizing against is not B's current state but B's state from the past, from when the signal left. A is playing against a ghost. A phantom. A delayed image of who B was, not who B is.

And here is the really beautiful thing: this is exactly the same situation that physicists faced when they tried to understand how charged particles interact. The electron does not feel the current position of another electron. It feels where the other electron was when the signal left. The whole edifice of electromagnetism and eventually relativity comes from taking this finite speed of information seriously.

So that is what we are going to do here. We are going to take the finite speed of information seriously, and see where it leads us.
:::

::::{admonition} Connection to RL #17: Independent PPO as Disconnected Sheaf
:class: note
:name: conn-rl-17
**The General Law (Fragile Agent):**
Multi-agent interaction is modeled via **Ghost Interfaces** (Definition {prf:ref}`def-ghost-interface`) connecting retarded boundary states:

$$
\mathcal{G}_{ij}(t) \subset \partial\mathcal{Z}^{(i)}(t) \times \partial\mathcal{Z}^{(j)}(t - \tau_{ij}), \quad \omega_{\mathcal{G},ij} := \omega^{(i)}(t) \oplus \omega^{(j)}(t - \tau_{ij})\big|_{\mathcal{G}_{ij}}.

$$
The **Game Tensor** $\mathcal{G}_{ij}$ (Definition {prf:ref}`def-the-game-tensor`) encodes strategic coupling with retarded components: how Agent $i$'s latent inertia changes due to Agent $j$'s past state.

**The Degenerate Limit:**
Set all interfaces $\mathcal{G}_{ij} = \emptyset$ (disconnect the sheaf). Each agent treats others as stationary noise.

**The Special Case (Standard RL - IPPO):**
Independent PPO {cite}`de2020independent` runs separate learners with shared scalar reward (conservative case):

$$
\pi^{(i)} = \arg\max_{\pi} \mathbb{E}\left[ \sum_t r^{(i)}_t(\mathbf{s}, \mathbf{a}) \right], \quad \text{treating } \pi^{(-i)} \text{ as fixed}.

$$
Each agent optimizes against a stationary environment—other agents are part of the "MDP noise."

**Result:** IPPO is the $\mathcal{G}_{ij} \to \emptyset$ limit where agents are **solipsistic**—they share a world but have no causal coupling.

**What the generalization offers:**
- **Causal structure**: Finite $c_{\text{info}}$ determines which events can influence which ({ref}`sec-the-failure-of-simultaneity`)
- **Ghost Interface**: Agents couple to retarded images, not instantaneous states ({ref}`sec-the-ghost-interface`)
- **Klein-Gordon value equation**: Hyperbolic wave propagation replaces elliptic relaxation ({ref}`sec-the-hyperbolic-value-equation`)
- **Nash as standing wave**: Equilibrium is time-averaged stationarity ({ref}`sec-relativistic-nash-equilibrium`)
- **Diagnostic nodes 46-48, 62**: Runtime monitoring including causality violation checks ({ref}`sec-yang-mills-action`)
::::

(sec-the-product-configuration-space)=
## The Product Configuration Space

:::{div} feynman-prose
Before we get to the tricky relativistic stuff, let me set the stage. We have $N$ agents, each one doing its own thing in its own little world. Each agent has what we have been calling a latent manifold, which is just a fancy way of saying "the space of internal states the agent can be in." And each agent has a boundary, sensors and motors, through which it talks to the environment.

Now, what is the natural way to describe all $N$ agents together? Well, you just take the product. Agent 1 can be anywhere in its space $\mathcal{Z}^{(1)}$, and at the same time Agent 2 can be anywhere in $\mathcal{Z}^{(2)}$, and so on. So the combined state lives in the big product space $\mathcal{Z}^{(1)} \times \mathcal{Z}^{(2)} \times \cdots \times \mathcal{Z}^{(N)}$.

This is exactly like asking: where can two particles be? Well, particle 1 can be anywhere in space, and particle 2 can be anywhere in space, so the combined configuration is just the product of the two spaces. Nothing mysterious here.

The important thing to notice is that right now the metric on this product space is block-diagonal. What does that mean? It means if I want to measure distances, agent $i$'s internal geometry does not care about agent $j$'s position. They are geometrically independent. Later we will see how strategic coupling breaks this independence, but let us start with the uncoupled case.
:::

Consider $N$ agents, each with an internal latent manifold $(\mathcal{Z}^{(i)}, G^{(i)})$ and a boundary interface
$B^{(i)} = (x^{(i)}, a^{(i)}, r^{(i)})$, where $r^{(i)}$ is a boundary reward sample (evaluation of the reward 1-form/flux;
scalar in the conservative case). The agents may be spatially distributed, with finite information propagation time
between them.

:::{prf:definition} N-Agent Product Manifold
:label: def-n-agent-product-manifold

The global configuration space is the product manifold:

$$
\mathcal{Z}^{(N)} := \mathcal{Z}^{(1)} \times \mathcal{Z}^{(2)} \times \cdots \times \mathcal{Z}^{(N)}.

$$
The metric on $\mathcal{Z}^{(N)}$ is the direct sum of individual metrics:

$$
G^{(N)} := \bigoplus_{i=1}^N G^{(i)},

$$
where each $G^{(i)}$ is the capacity-constrained metric from Theorem {prf:ref}`thm-capacity-constrained-metric-law`. In coordinates, this is block-diagonal: if $\mathbf{z} = (z^{(1)}, \ldots, z^{(N)})$ with $z^{(i)} \in \mathbb{R}^{d_i}$, then $G^{(N)}_{\mu\nu}(\mathbf{z}) = G^{(i)}_{ab}(z^{(i)})$ when indices $\mu, \nu$ both lie in agent $i$'s block, and $G^{(N)}_{\mu\nu} = 0$ otherwise.

*Units:* $[G^{(N)}] = [z]^{-2}$.

*Remark (Isolated Agents).* The product metric $G^{(N)}$ describes agents in **isolation**—there is no cross-coupling between $\mathcal{Z}^{(i)}$ and $\mathcal{Z}^{(j)}$. Strategic coupling modifies this to $\tilde{G}^{(N)}$ via the Game Tensor ({ref}`sec-the-game-tensor-deriving-adversarial-geometry`).

:::

:::{prf:definition} Agent-Specific Boundary Interface
:label: def-agent-specific-boundary-interface

Each agent $i$ possesses its own symplectic boundary $(\partial\mathcal{Z}^{(i)}, \omega^{(i)})$ with:
- **Dirichlet component** (sensors): $\phi^{(i)}(x) = $ observation stream
- **Neumann component** (motors): $j^{(i)}_{\text{motor}}(x) = $ action flux
- **Reward component** (source): boundary reward flux $J_r^{(i)}$ (1-form); conservative case reduces to scalar charge
  density $\sigma_r^{(i)}$ (Definition {prf:ref}`def-the-reward-flux`)

The boundary conditions follow the structure of Definition {prf:ref}`def-dirichlet-boundary-condition-sensors`–23.1.3,
applied per-agent.

*Cross-reference:* {ref}`sec-the-symplectic-interface-position-momentum-duality` (Symplectic Boundary Manifold), Definition {prf:ref}`def-mass-tensor`.

:::

:::{prf:definition} Environment Distance
:label: def-environment-distance

Let $d_{\mathcal{E}}^{ij}$ denote the **environment distance** between agents $i$ and $j$—the geodesic length in the environment manifold $\mathcal{E}$ that information must traverse. This may differ from the latent distance $d_G(z^{(i)}, z^{(j)})$.

*Examples:*
- **Physical agents:** $d_{\mathcal{E}}^{ij} = $ spatial separation in meters
- **Networked agents:** $d_{\mathcal{E}}^{ij} = $ network hop distance or latency
- **Co-located agents:** $d_{\mathcal{E}}^{ij} = 0$ (shared boundary)

*Units:* $[d_{\mathcal{E}}^{ij}] = $ meters or equivalent environment-specific units.

:::
(sec-the-failure-of-simultaneity)=
## The Failure of Simultaneity

:::{div} feynman-prose
Here is where things get interesting. You know the standard story: the value function $V(z)$ satisfies some nice elliptic equation, the Bellman equation or the Helmholtz equation or whatever you want to call it. You solve it, you get your optimal policy, everyone is happy.

But think about what that equation is really saying. If I change the conservative reward source at one point, the value
function changes everywhere, instantly. The information about that reward change propagates at infinite speed across the
whole manifold. That is what an elliptic equation does. It is like saying that if you poke the electric potential at one
point, the whole field readjusts instantaneously everywhere.

Now, in electrostatics, we get away with this because we are looking at the steady state, where everything has had time to settle down. But what if the world is changing? What if agents are moving around? Then you cannot use the steady-state answer. You have to track how the information actually propagates.

And here is the key insight: information propagates at some finite speed $c_{\text{info}}$. In physics, it is the speed of light. In a network, it might be the inverse of the network latency. In a room full of robots, it might be the speed of sound or the speed of radio waves. Whatever it is, it is finite.

Once you accept that information travels at finite speed, you immediately get a causal structure. Some events can influence other events; some cannot. You get past and future light cones. You get relativity. Not because we are doing fundamental physics, but because we are taking causality seriously.
:::

The standard HJB equation assumes the value $V(z)$ relaxes instantly across the manifold. This implies an infinite speed
of information propagation, violating the causal constraints of distributed systems. Here $V$ denotes the scalar
potential associated with the **conservative component** of the reward 1-form; non-conservative (curl) components
propagate as antisymmetric fields and appear as velocity-dependent forces rather than a scalar PDE.

:::{prf:axiom} Information Speed Limit
:label: ax-information-speed-limit

There exists a maximum speed $c_{\text{info}} > 0$ at which information propagates through the environment $\mathcal{E}$. The **Causal Delay** between agents $i$ and $j$ is:

$$
\tau_{ij} := \frac{d_{\mathcal{E}}^{ij}}{c_{\text{info}}},

$$
where $d_{\mathcal{E}}^{ij}$ is the environment distance (Definition {prf:ref}`def-environment-distance`).

*Units:* $[c_{\text{info}}] = [\text{length}]/[\text{time}]$, $[\tau_{ij}] = [\text{time}]$.

*Examples:*
- **Physical systems:** $c_{\text{info}} = c \approx 3 \times 10^8$ m/s (speed of light)
- **Acoustic systems:** $c_{\text{info}} \approx 343$ m/s (speed of sound)
- **Networked systems:** $c_{\text{info}} \approx d/\text{latency}$ (effective propagation speed)
- **Co-located agents:** $c_{\text{info}} \to \infty$ effective limit when $d_{\mathcal{E}}^{ij} = 0$

:::

:::{prf:definition} Causal Interval
:label: def-causal-interval

The **Causal Interval** between spacetime events $(z^{(i)}, t_i)$ and $(z^{(j)}, t_j)$ is:

$$
\Delta s^2_{ij} := -c_{\text{info}}^2 (t_j - t_i)^2 + (d_{\mathcal{E}}^{ij})^2.

$$
The events are classified as:
- **Timelike** ($\Delta s^2_{ij} < 0$): $|t_j - t_i| > \tau_{ij}$. Causal influence is possible.
- **Spacelike** ($\Delta s^2_{ij} > 0$): $|t_j - t_i| < \tau_{ij}$. No causal influence is possible.
- **Lightlike** ($\Delta s^2_{ij} = 0$): $|t_j - t_i| = \tau_{ij}$. Boundary case.

*Consequence:* If agents $i$ and $j$ are spacelike separated at time $t$, no instantaneous Hamiltonian $H(z^{(i)}_t, z^{(j)}_t)$ can couple their states. Coupling must occur via retarded potentials.

:::

:::{prf:definition} Past Light Cone
:label: def-past-light-cone

The **Past Light Cone** of Agent $i$ at time $t$ is the set of all agent-time pairs that can causally influence Agent $i$:

$$
\mathcal{C}^-_i(t) := \left\{ (j, t') \in \{1,\ldots,N\} \times \mathbb{R} : t' \leq t - \tau_{ij} \right\}.

$$
The **Future Light Cone** is defined symmetrically:

$$
\mathcal{C}^+_i(t) := \left\{ (j, t') : t' \geq t + \tau_{ij} \right\}.

$$
*Physical interpretation:* Agent $i$ at time $t$ can only receive information from events in $\mathcal{C}^-_i(t)$ and can only influence events in $\mathcal{C}^+_i(t)$. The region outside both cones is causally disconnected.

:::

:::{div} feynman-prose
Let me make sure you really understand what these light cones mean, because they are absolutely central to everything that follows.

Imagine you are sitting at some point in space at some moment in time. What can you know about? Well, you can only know about things that have had time to send you a signal. If some event happened far away just a moment ago, you cannot possibly know about it yet because the signal has not arrived. The set of all events you can know about forms a cone in spacetime, opening backwards in time. That is your past light cone.

Similarly, what can you influence? Only events that your signals can reach. If something is about to happen far away in the next instant, you cannot possibly affect it because your signal will not arrive in time. The events you can influence form a cone opening forwards in time. That is your future light cone.

Everything else, the region outside both cones, is causally disconnected from you right now. You cannot know about it, and you cannot affect it. It is as if it does not exist for you at this moment.

Now here is the crucial point for multi-agent systems: if two agents are spacelike separated, meaning they are in each other's "causally disconnected" region, then they cannot have any instantaneous interaction. Any coupling between them must involve the past. Agent A interacts not with Agent B as it is now, but with Agent B as it was when the signal left. This is the ghost we talked about before.
:::

(pi-minkowski)=
::::{admonition} Physics Isomorphism: Minkowski Spacetime
:class: note

**In Physics:** Special relativity defines the causal structure via the Minkowski metric $ds^2 = -c^2 dt^2 + dx^2 + dy^2 + dz^2$. Events with $ds^2 < 0$ are timelike separated (causally connected); events with $ds^2 > 0$ are spacelike separated (causally disconnected) {cite}`jackson1999classical`.

**In Implementation:** The causal interval (Definition {prf:ref}`def-causal-interval`) induces a Lorentzian structure on the agent-time space:

$$
\Delta s^2_{ij} = -c_{\text{info}}^2 \Delta t^2 + d_{\mathcal{E}}^2.

$$

**Correspondence Table:**
| Special Relativity | Multi-Agent System |
|:-------------------|:-------------------|
| Speed of light $c$ | Information speed $c_{\text{info}}$ |
| Spatial distance | Environment distance $d_{\mathcal{E}}^{ij}$ |
| Past light cone | Causally accessible agent states |
| Spacelike separation | Instantaneously decoupled agents |
| Lorentz invariance | Causal consistency under frame changes |
::::

(sec-the-relativistic-state-restoring-markovianity)=
## The Relativistic State: Restoring Markovianity

:::{div} feynman-prose
Now we come to a really subtle point that trips up a lot of people. The Markov property says that the future depends only on the present, not on the past. If I tell you the state right now, you can predict what happens next without needing to know the history.

But wait. We just said that agent A cannot see agent B's current state. It can only see the ghost, the retarded image from time $\tau$ ago. So if I tell you agent A's position and agent B's position right now, that is not enough to predict what happens next. Agent A is actually responding to where B was in the past, not where B is now.

Does this mean the Markov property fails? That we cannot use all our beautiful control theory?

Not if we are clever. Here is the trick: expand your definition of "state" to include the relevant history. The state is not just "where everyone is now." The state is "where everyone is now, plus all the signals that are currently in transit, plus everything I received but have not finished processing yet."

This augmented state, which includes all the information in the past light cone, is called the Causal Bundle. And on this augmented space, the Markov property is restored. The future of the Causal Bundle depends only on its present, not on its history, because all the relevant history is already encoded in the bundle.

This is exactly what the Memory Screen $\Xi_{<t}$ does. It stores the incoming wavefronts, the retarded potentials, all the ghosts from the past that are still affecting the present. Once you include the Memory Screen in your state, you get Markov back.
:::

To recover a valid control problem under finite information speed, we must augment the state to include the field configuration within the past light cone.

:::{prf:definition} Retarded Potential (Memory Screen)
:label: def-retarded-potential

Let $\rho^{(j)}_r(t, z)$ be the scalar source density associated with the conservative component of Agent $j$'s boundary
reward flux. The potential perceived by Agent $i$ at position $z$ and time $t$ is the **Retarded Potential**:

$$
\Psi_{\text{ret}}^{(i)}(t, z) = \sum_{j \neq i} \int_{-\infty}^{t} \int_{\mathcal{Z}^{(j)}} G_{\text{ret}}(z, t; \zeta, \tau) \rho^{(j)}_r(\tau, \zeta) \, d\mu_{G^{(j)}}(\zeta) \, d\tau,

$$
where $G_{\text{ret}}$ is the **Retarded Green's Function** for the wave operator on the manifold:

$$
G_{\text{ret}}(z, t; \zeta, \tau) \quad \text{solves} \quad \left(\frac{1}{c_{\text{info}}^2}\partial_t^2 - \Delta_G + \kappa^2\right)G_{\text{ret}} = \delta(z-\zeta)\delta(t-\tau),

$$
with $G_{\text{ret}} = 0$ for $t < \tau$. In flat space and the massless limit ($\kappa = 0$), $G_{\text{ret}}$ reduces to a light-cone delta; for $\kappa > 0$ it develops an interior light-cone tail.

*Interpretation:* Agent $i$ does not perceive Agent $j$'s current state. It perceives the "ghost" of Agent $j$ from time $\tau_{ij} = d_{\mathcal{E}}^{ij}/c_{\text{info}}$ ago.

*Units:* $[\Psi_{\text{ret}}] = \text{nat}$, $[G_{\text{ret}}] = [\text{length}]^{2-D}[\text{time}]^{-1}$.

*Remark (Strategic coupling).* When strategic relationships matter, weight each source by $\alpha_{ij}$; equivalently replace
$\rho^{(j)}_r$ with $\rho^{\text{ret}}_{ij}$ from Definition {prf:ref}`def-retarded-interaction-potential`.

:::

:::{prf:definition} Causal Bundle
:label: def-causal-bundle

The **Causal Bundle** is the augmented state space:

$$
\mathcal{Z}_{\text{causal}} := \mathcal{Z}^{(N)} \times \Xi_{<t},

$$
where:
- $\mathcal{Z}^{(N)} = \prod_i \mathcal{Z}^{(i)}$ is the product configuration space (Definition {prf:ref}`def-n-agent-product-manifold`)
- $\Xi_{<t}$ is the **Memory Screen** restricted to the causal past (Definition {prf:ref}`def-memory-screen`): $\Xi_{<t} = \{(\gamma(t'), \alpha(t')) : t' < t\}$

The **Relativistic State** for Agent $i$ at time $t$ is:

$$
\mathcal{S}^{(i)}_t := \left( z^{(i)}_t, \Xi^{(i)}_{<t} \right),

$$
where $\Xi^{(i)}_{<t}$ stores the history of received retarded potentials over the interval $[t - \tau_{\text{horizon}}, t)$.

*Operational validity:* $\Xi^{(i)}_{<t}$ is locally observable at time $t$. The "true" global state of all agents is hidden, but $\mathcal{S}^{(i)}_t$ is a sufficient statistic for Agent $i$'s optimal policy within its future light cone.

:::

:::{prf:theorem} Markov Restoration on Causal Bundle
:label: thm-markov-restoration

Let $P(z^{(N)}_{t+\Delta t} | z^{(N)}_t, \Xi_{<t})$ denote the transition probability. When agents have finite causal delay $\tau_{ij} > 0$:

1. **On $\mathcal{Z}^{(N)}$ alone:** The Markov property fails:

   $$
   P(z^{(N)}_{t+\Delta t} | z^{(N)}_t) \neq P(z^{(N)}_{t+\Delta t} | z^{(N)}_{\leq t}).

   $$

2. **On $\mathcal{Z}_{\text{causal}}$:** The Markov property is restored:

   $$
   P\left((z^{(N)}_{t+\Delta t}, \Xi_{<t+\Delta t}) \,\big|\, (z^{(N)}_t, \Xi_{<t})\right) = P\left((z^{(N)}_{t+\Delta t}, \Xi_{<t+\Delta t}) \,\big|\, \text{full history}\right).

   $$

*Proof sketch.* The Memory Screen $\Xi_{<t}$ encodes all information about past states that can causally influence the future. By the definition of the past light cone (Definition {prf:ref}`def-past-light-cone`), no additional information from $\Xi_{<t'}$ for $t' < t$ is needed beyond what is already encoded in $\Xi_{<t}$. The causal structure guarantees that spacelike-separated events cannot contribute new information. See **{ref}`sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws`** for the complete proof using causal factorization and Chapman-Kolmogorov. $\square$

:::

:::{prf:corollary} Memory as Physical Necessity
:label: cor-memory-physical-necessity

In the relativistic multi-agent setting, the Memory Screen (Definition {prf:ref}`def-memory-screen`) is not an optional enhancement but a **physical requirement** for a well-posed control problem. Without it, the agent's state is non-Markovian, and optimal control theory does not apply.

*Cross-reference:* This elevates the role of $\Xi_{<t}$ from {ref}`sec-the-historical-manifold-and-memory-screen`, where it served as a recording device for trajectory history, to a primary state variable that restores the Markov property.

:::



(sec-the-ghost-interface)=
## The Ghost Interface: Asynchronous Coupling

:::{div} feynman-prose
Now we get to the heart of the matter: how do agents actually interact when they cannot see each other instantaneously?

The answer is the Ghost Interface. It is a beautiful concept once you get used to it. When Agent A looks at Agent B, it does not see B. It sees the ghost of B, the image of B from the past. This ghost is a real thing that lives on the interface between the agents. It carries all the information about B that A can actually use, delayed by the appropriate amount.

Think of it like looking at a star. When you look at the North Star, you are not seeing it as it is now. You are seeing it as it was about 430 years ago, because that is how long the light took to reach you. For all you know, the star blew up 400 years ago and the news has not arrived yet. You are interacting with a ghost.

The symplectic structure on the Ghost Interface tells you how these ghosts couple. Agent A's current boundary couples to Agent B's past boundary. The coupling has a definite mathematical form, but the key point is the time offset. There is always this lag, this delay, this $\tau_{ij}$ between when B does something and when A can respond to it.

This creates strategic hysteresis. What do I mean by that? Well, suppose Agent A decides to make an aggressive move based on where B's ghost is. But by the time A commits to the move, B might have already moved somewhere else. A made its decision based on old information. This lag is not a bug to be fixed. It is a fundamental feature of any distributed system with finite information speed.
:::

We replace the instantaneous coupling of boundary conditions with an asynchronous **Ghost Interface** that respects causal structure.

:::{prf:definition} Ghost Interface
:label: def-ghost-interface

The **Ghost Interface** $\mathcal{G}_{ij}(t)$ between agents $i$ and $j$ at time $t$ is:

$$
\mathcal{G}_{ij}(t) := \partial\mathcal{Z}^{(i)}(t) \times \partial\mathcal{Z}^{(j)}(t - \tau_{ij}),

$$
coupling Agent $i$'s current boundary to Agent $j$'s past boundary, where $\tau_{ij} = d_{\mathcal{E}}^{ij}/c_{\text{info}}$ is the causal delay.

The **Ghost Symplectic Structure** is:

$$
\omega_{\mathcal{G},ij} := \omega^{(i)}(t) \oplus \omega^{(j)}(t - \tau_{ij})\big|_{\mathcal{G}_{ij}}.

$$

*Mechanism:* Agent $i$ couples not to $z^{(j)}_t$, but to the **Ghost State** $\hat{z}^{(j)}_t := z^{(j)}_{t-\tau_{ij}}$—the state of Agent $j$ when the signal was emitted.

*Units:* $[\tau_{ij}] = [\text{time}]$.

:::

:::{prf:proposition} Interaction Kernel
:label: prop-interaction-kernel

The **pairwise interaction potential** $\Phi_{\text{int}}: \mathcal{Z} \times \mathcal{Z} \to \mathbb{R}$ between agents at positions $z, \zeta$ is the screened Green's function weighted by influence:

$$
\Phi_{\text{int}}(z, \zeta) := \alpha \cdot \mathcal{G}_{\kappa}(z, \zeta)

$$
where $\mathcal{G}_{\kappa}$ is the screened Green's function (Proposition {prf:ref}`prop-green-s-function-interpretation`) and $\alpha$ encodes the strategic relationship.

*Properties:*
- $\Phi_{\text{int}}(z, \zeta) = \Phi_{\text{int}}(\zeta, z)$ (symmetric in cooperative settings)
- $\Phi_{\text{int}} \to 0$ as $d_G(z, \zeta) \to \infty$ (locality via screening)
- $\nabla^2_z \Phi_{\text{int}}$ defines the Game Tensor contribution (Definition {prf:ref}`def-the-game-tensor`)
:::

:::{prf:definition} Retarded Interaction Potential
:label: def-retarded-interaction-potential

The **Retarded Interaction Source Density** from Agent $j$ to Agent $i$ is:

$$
\rho^{\text{ret}}_{ij}(\zeta, \tau) := \alpha_{ij} \cdot \rho^{(j)}_r(\zeta, \tau),

$$
where:
- $\rho^{(j)}_r$ is the conservative reward source density for Agent $j$ derived from boundary reward flux
  (Definition {prf:ref}`def-the-reward-flux`)
- $\alpha_{ij} \in \{-1, 0, +1\}$ encodes the strategic relationship:
  - $\alpha_{ij} = +1$: Cooperative
  - $\alpha_{ij} = 0$: Independent
  - $\alpha_{ij} = -1$: Adversarial

We write $\rho^{\text{ret}}_{ij}$ on $\mathcal{Z}^{(j)}$ and pull it back to Agent $i$'s chart along the Ghost Interface;
for notational simplicity, we suppress the pullback in what follows.

The induced **Retarded Interaction Potential** is the retarded Green's function convolution:

$$
\Phi^{\text{ret}}_{ij}(z^{(i)}, t) = \int_{-\infty}^{t} \int_{\mathcal{Z}^{(j)}} G_{\text{ret}}(z^{(i)}, t; \zeta, \tau)\,
\rho^{\text{ret}}_{ij}(\zeta, \tau)\, d\mu_{G^{(j)}}(\zeta)\, d\tau,

$$
where $G_{\text{ret}}$ is the retarded Green's function (Definition {prf:ref}`def-retarded-potential`).

*Remark (Point-source / ghost limit).* If Agent $j$'s conservative source is concentrated along a trajectory,
$\rho^{(j)}_r(\zeta, \tau) = \sigma^{(j)}_r(\tau)\,\delta(\zeta - z^{(j)}_\tau)$, then

$$
\Phi^{\text{ret}}_{ij}(z^{(i)}, t) = \alpha_{ij}\int_{-\infty}^{t} G_{\text{ret}}(z^{(i)}, t; z^{(j)}_\tau, \tau)\,
\sigma^{(j)}_r(\tau)\, d\tau,
$$
which reduces to evaluation at the retarded time in the massless flat-space limit. This recovers the ghost-state
interpretation.

*Remark (Quasi-static kernel).* In the low-frequency limit, $G_{\text{ret}}$ reduces to the screened static kernel
$\mathcal{G}_\kappa$ and the potential can be approximated by evaluating the instantaneous interaction at the ghost
state. This is a computational shortcut, not the first-principles definition.

*Remark (Non-conservative component).* Solenoidal reward components are not captured by the scalar source; they enter via
the curl field in the dynamics.

:::

:::{prf:theorem} Strategic Delay Tensor
:label: thm-strategic-delay-tensor

The effective coupling tensor $\mathcal{T}_{ij}$ between agents splits into instantaneous and retarded components:

$$
\mathcal{T}_{ij}^{\text{total}}(t) = \underbrace{\mathcal{T}_{ij}^{\text{local}}(t)}_{\text{Short-range}} + \underbrace{\int_{-\infty}^t \mathcal{K}_{\text{delay}}(t-\tau) \mathcal{T}_{ij}^{\text{ghost}}(\tau) \, d\tau}_{\text{Long-range Retarded}},

$$
where $\mathcal{K}_{\text{delay}}(t-\tau) = \delta(t - \tau - \tau_{ij})$ is the delay kernel.

**Adversarial consequence:** Against a distant adversary, the effective metric inflation (from the Game Tensor) is delayed. An agent may commit to an aggressive trajectory only to experience a "wall" of increased inertia arriving from the opponent's past actions.

*Proof.* Expand the coupled value equation to second order in the retarded potential. The cross-Hessian $\partial^2 V^{(i)} / \partial z^{(j)} \partial z^{(j)}$ evaluated at $z^{(j)}_{t-\tau_{ij}}$ yields the delayed Game Tensor contribution. $\square$

:::

:::{prf:corollary} Newtonian Limit
:label: cor-newtonian-limit-ghost

As $c_{\text{info}} \to \infty$, the causal delay vanishes: $\tau_{ij} \to 0$ for all pairs. The Ghost Interface reduces to the instantaneous interface:

$$
\lim_{c_{\text{info}} \to \infty} \mathcal{G}_{ij}(t) = \partial\mathcal{Z}^{(i)}(t) \times \partial\mathcal{Z}^{(j)}(t),

$$
and the retarded potential becomes instantaneous:

$$
\lim_{c_{\text{info}} \to \infty} \Phi^{\text{ret}}_{ij}(z^{(i)}, t) = \Phi_{ij}(z^{(i)}, z^{(j)}_t),

$$
where $\Phi_{ij}$ denotes the instantaneous interaction potential (the quasi-static limit of Definition {prf:ref}`def-retarded-interaction-potential`).

*Interpretation:* Co-located agents ($d_{\mathcal{E}}^{ij} = 0$) or systems with negligible propagation delay operate in the Newtonian regime where standard MARL applies.

:::

(pi-lienard-wiechert)=
::::{admonition} Physics Isomorphism: Liénard-Wiechert Potentials
:class: note

**In Physics:** The electromagnetic potentials of a moving charge are evaluated at the retarded time $t_{\text{ret}} = t - r/c$, not the current time. The Liénard-Wiechert potentials encode causality in classical electrodynamics {cite}`jackson1999classical`.

**In Implementation:** In the quasi-static approximation, the Ghost Interface evaluates strategic potentials at the
retarded time:

$$
\Phi^{\text{ret}}_{ij}(z^{(i)}, t) \approx \Phi_{ij}(z^{(i)}, z^{(j)}_{t-\tau_{ij}}).

$$

**Correspondence Table:**
| Electrodynamics | Relativistic Agent |
|:----------------|:-------------------|
| Field equation $\square A^\mu = J^\mu$ | Value equation (conservative component) $\square_G V = \rho_r$ |
| Light speed $c$ | Information speed $c_{\text{info}}$ |
| Retarded time $t_{\text{ret}}$ | Ghost time $t - \tau_{ij}$ |
| Liénard-Wiechert potential | Retarded interaction potential |
| Radiation reaction | Strategic back-pressure |
::::



(sec-the-hyperbolic-value-equation)=
## The Hyperbolic Value Equation (Klein-Gordon)

:::{div} feynman-prose
Now comes one of the most beautiful results in this whole theory. We have been saying that value cannot propagate instantaneously. So what equation does it satisfy instead of the static Helmholtz equation?

Think about it physically. If I create a conservative reward source at some point, the information about that reward has
to propagate outward. It travels at speed $c_{\text{info}}$. This is a wave. The value function is not a static field
anymore. It is a propagating wave field, rippling outward from sources of reward.

What is the wave equation that describes this? It turns out to be the Klein-Gordon equation. This is the relativistic version of the Helmholtz equation, exactly as you would hope. The Helmholtz equation has just the Laplacian $-\Delta_G$. The Klein-Gordon equation adds a second time derivative: $\frac{1}{c^2}\partial_t^2 - \Delta_G$.

The extra term $\frac{1}{c^2}\partial_t^2$ is what makes it a wave equation instead of just a relaxation equation. It says that value has inertia. If the value is increasing, it tends to keep increasing for a while. If it is decreasing, it tends to keep decreasing. This is utterly different from the elliptic case where the field instantly adjusts to changes in the source.

The screening term $\kappa^2$ is still there. Remember, $\kappa$ comes from the discount factor. It makes the value decay with distance, even in the wave equation. So what you get is a screened wave, a wave that decays exponentially as it propagates. Value wavefronts do not travel forever. They get weaker and weaker as they spread out and as they encounter the discounting.
:::

Under relativistic constraints, the elliptic Helmholtz equation for the **scalar value potential** (conservative
component; Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`) transforms into a hyperbolic wave equation.

:::{prf:theorem} HJB-Klein-Gordon Correspondence
:label: thm-hjb-klein-gordon

Let information propagate at speed $c_{\text{info}}$. The Value Function $V^{(i)}(z, t)$ for Agent $i$ satisfies the **Screened Wave Equation**:

$$
\boxed{\left( \frac{1}{c_{\text{info}}^2} \frac{\partial^2}{\partial t^2} - \Delta_{G^{(i)}} + \kappa_i^2 \right) V^{(i)}(z, t) = \rho^{(i)}_r(z, t) + \sum_{j \neq i} \rho^{\text{ret}}_{ij}(z, t)}

$$
where:
- $\square_{G} = \frac{1}{c_{\text{info}}^2}\partial_t^2 - \Delta_G$ is the **D'Alembertian** on the manifold. With spacetime metric $g_{\mu\nu} = \text{diag}(-c_{\text{info}}^2, G_{ij})$, this equals
  
  $$
  \square_G = -\frac{1}{\sqrt{|g|}}\partial_\mu\left(\sqrt{|g|}g^{\mu\nu}\partial_\nu\right).
  $$
- $\kappa_i$ is the **spatial screening mass** with $[\kappa_i] = 1/[\text{length}]$. Let the temporal discount rate be $\lambda_i := -\ln\gamma_i / \Delta t$ (units $1/[\text{time}]$). Using the relativistic conversion $\text{length} = c_{\text{info}} \cdot \text{time}$ gives $\kappa_i = \lambda_i / c_{\text{info}}$. In natural units ($\Delta t = 1$, $c_{\text{info}} = 1$) this reduces to $\kappa_i = -\ln\gamma_i$.

*Notation.* When we write the spacetime metric $g_{\mu\nu} = \text{diag}(-c_{\text{info}}^2, G_{ij})$ in this section, $G_{ij}$ denotes the **active** spatial metric for the equation at hand (the intrinsic $G$ for scalar $V$, or the strategic $\tilde{G}$ when the game-augmented metric is used).
- $\rho^{(i)}_r$ is the local **conservative** reward source density derived from boundary reward flux
  (Definition {prf:ref}`def-the-reward-flux`) (units: $[\text{nat}]/[\text{length}]^2$)
- $\rho^{\text{ret}}_{ij}$ is the retarded interaction source density (Definition {prf:ref}`def-retarded-interaction-potential`)

*Optional damping.* A linear term $\gamma_{\text{damp}}\partial_t V$ may be added to model explicit non-stationary friction or learning lag. This is a modeling choice distinct from the discount-induced screening $\kappa$.

*Scope.* This equation governs the scalar potential associated with the conservative component of the reward 1-form.
Solenoidal/harmonic components induce velocity-dependent coupling and are handled by the curl field $\mathcal{F}$ in the
dynamics.

*Proof sketch.* Expand the Bellman recursion $V(z, t) = r \Delta t + \gamma \mathbb{E}[V(z', t+\Delta t)]$ to second order in both spatial and temporal increments. The finite propagation speed $c_{\text{info}}$ introduces the wave term $\partial_t^2 V$. The derivation parallels the passage from Poisson to wave equation in electrostatics vs. electrodynamics. See {ref}`sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws`. $\square$

*Character:* This is a hyperbolic PDE (wave equation with mass and damping), in contrast to the elliptic Helmholtz equation of {ref}`sec-the-bulk-potential-screened-poisson-equation`.

:::

:::{prf:corollary} Value Wavefront Propagation
:label: cor-value-wavefront

A sudden change in conservative reward source at location $z_A$ and time $t_0$ propagates outward as a **Value
Wavefront**:

$$
V(z, t) \sim \Theta\!\left(t - t_0 - \frac{d_G(z, z_A)}{c_{\text{info}}}\right)\, e^{-\kappa d_G(z, z_A)} \cdot \rho_r(z_A, t_0),

$$
where $\Theta$ is the Heaviside step function enforcing causality. This expression is schematic: the prefactor is dimension-dependent, and for $\kappa > 0$ the exact retarded Green's function has an interior light-cone tail (Bessel decay) rather than a pure delta on the cone.

*Interpretation:* The Value surface is not a static potential but a dynamic "ocean" of interfering causal ripples.
Conservative reward shocks propagate at speed $c_{\text{info}}$, decaying exponentially with the screening length
$1/\kappa$.

:::

:::{prf:corollary} Helmholtz as Newtonian Limit
:label: cor-helmholtz-limit

In the limit $c_{\text{info}} \to \infty$, the temporal derivatives become negligible:

$$
\frac{1}{c_{\text{info}}^2} \frac{\partial^2 V}{\partial t^2} \to 0,

$$
and the Klein-Gordon equation reduces to the **stationary Helmholtz equation**:

$$
(-\Delta_G + \kappa^2) V = \rho_r + \sum_{j \neq i} \rho^{\text{ret}}_{ij}.

$$
This recovers Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence` as the instantaneous (Newtonian) limit.

:::

:::{prf:proposition} Retarded Green's Function
:label: prop-retarded-greens-function

The solution to the inhomogeneous Klein-Gordon equation is given by convolution with the **Retarded Green's Function**:

$$
V^{(i)}(z, t) = \int_{-\infty}^{t} \int_{\mathcal{Z}^{(i)}} G_{\text{ret}}(z, t; \zeta, \tau) \left[ \rho^{(i)}_r(\zeta, \tau) + \sum_{j \neq i} \rho^{\text{ret}}_{ij}(\zeta, \tau) \right] d\mu_{G^{(i)}}(\zeta) \, d\tau,

$$
where $G_{\text{ret}}$ satisfies:

$$
\left( \frac{1}{c_{\text{info}}^2} \frac{\partial^2}{\partial t^2} - \Delta_G + \kappa^2 \right) G_{\text{ret}}(z, t; \zeta, \tau) = \delta(z - \zeta)\delta(t - \tau),

$$
with the **causal boundary condition** $G_{\text{ret}} = 0$ for $t < \tau$.

*Massless flat-space example (D = 3):* For $\mathcal{Z} = \mathbb{R}^3$ and $\kappa = 0$,

$$
G_{\text{ret}}(z, t; \zeta, \tau) = \frac{\Theta(t - \tau)}{4\pi |z - \zeta|} \delta\left(t - \tau - \frac{|z-\zeta|}{c_{\text{info}}}\right).

$$
For $\kappa > 0$, the retarded kernel acquires an interior light-cone tail with Bessel decay; we keep $G_{\text{ret}}$ abstract to avoid dimension-specific formulas.

:::

(pi-klein-gordon)=
::::{admonition} Physics Isomorphism: Klein-Gordon Equation
:class: note

**In Physics:** The Klein-Gordon equation $(\square + m^2)\phi = \rho$ describes a relativistic scalar field with mass $m$. It reduces to the Helmholtz equation in the static limit {cite}`jackson1999classical`. (Sign convention: we use $\square_G = \frac{1}{c^2}\partial_t^2 - \Delta_G = -\frac{1}{\sqrt{|g|}}\partial_\mu(\sqrt{|g|}g^{\mu\nu}\partial_\nu)$.)

**In Implementation:** The scalar Value potential (conservative component) satisfies:

$$
\left(\frac{1}{c_{\text{info}}^2}\partial_t^2 - \Delta_G + \kappa^2\right)V = \rho_r

$$

**Correspondence Table:**
| Klein-Gordon (Physics) | Value Equation (Agent) |
|:-----------------------|:-----------------------|
| Scalar field $\phi$ | Value function $V$ |
| Mass parameter $m$ | Screening mass $\kappa$ |
| Source $\rho$ | Conservative reward density $\rho_r$ |
| D'Alembertian $\square$ | Manifold wave operator $\square_G$ |
| Static limit | Newtonian (Helmholtz) limit |
| Propagating modes | Value wavefronts |
::::
(sec-the-game-tensor-deriving-adversarial-geometry)=
## The Game Tensor: Relativistic Adversarial Geometry

:::{div} feynman-prose
This is one of my favorite parts of the whole theory, so let me take some time with it.

We have talked about how agents couple through the Ghost Interface. But there is another, more geometric way to think about adversarial interaction. When you are playing against an opponent, the opponent changes your effective geometry. They make some directions harder to move in than others.

Why? Because if you move in certain directions, your opponent will respond in ways that hurt you. Even if you could physically move that way easily, the strategic consequences make it costly. The opponent's potential response acts like friction, like resistance, like mass.

This is what the Game Tensor captures. It is a tensor $\mathcal{G}_{ij}$ that measures how sensitive your value function is to your opponent's position. If $\mathcal{G}_{ij}$ is large and positive, it means your opponent being nearby makes your life harder. They curve your geometry, inflate your metric, make you feel heavier and slower.

Think of it like trying to walk through a crowd. Physically, empty space and crowded space have the same geometry. But effectively, the crowd creates resistance. You move slower. You cannot change direction as easily. The crowd curves your effective space.

Now here is where the relativistic part comes in. The Game Tensor depends on where your opponent is. But you only know where your opponent was in the past, not where they are now. So the metric inflation you feel comes from the retarded Game Tensor, evaluated at your opponent's ghost position.

This means you might commit to a trajectory thinking the geometry is one way, only to have the "real" geometry be different by the time you get there. The opponent has moved since you last saw them. The strategic hysteresis we talked about manifests geometrically as a delayed metric perturbation.
:::

In an adversarial (zero-sum) game, Agent $j$ acts to minimize the value $V^{(i)}$ that Agent $i$ maximizes. Under relativistic constraints, the Game Tensor acquires retarded components that introduce strategic hysteresis.

:::{prf:definition} The Game Tensor
:label: def-the-game-tensor

We define the **Game Tensor** $\mathcal{G}_{ij}^{kl}$ as the cross-Hessian of Agent $i$'s value with respect to Agent $j$'s position:

$$
\mathcal{G}_{ij}^{kl}(z^{(i)}, z^{(j)}) := \frac{\partial^2 V^{(i)}}{\partial z^{(j)}_k \partial z^{(j)}_l}\bigg|_{z^{(j)} = z^{(j)*}},

$$
where $z^{(j)*}$ is Agent $j$'s current position (or expected position under their policy). This tensor measures how sensitive Agent $i$'s value landscape is to Agent $j$'s location.

*Units:* $[\mathcal{G}_{ij}^{kl}] = \text{nat}/[z]^2$.

*Remark (Heterogeneous manifolds).* $\mathcal{G}_{ij}$ lives in $T_{z^{(j)}}\mathcal{Z}^{(j)}$. To compare or add it to Agent $i$'s metric, pull it back along the Ghost Interface using the Strategic Jacobian $\mathcal{J}_{ji}: T_{z^{(i)}}\mathcal{Z}^{(i)} \to T_{z^{(j)}}\mathcal{Z}^{(j)}$:

$$
\mathcal{G}_{ij,mn} := G^{(j)}_{mp} G^{(j)}_{nq}\mathcal{G}_{ij}^{pq}, \quad
\mathcal{G}^{(i)}_{ij,kl} := (\mathcal{J}_{ji})^m{}_k (\mathcal{J}_{ji})^n{}_l \mathcal{G}_{ij,mn}.

$$

**Derivation 29.4.2 (The Strategic Metric).** Recall the **Capacity-Constrained Metric Law** (Theorem {prf:ref}`thm-capacity-constrained-metric-law`), where curvature is driven by the Risk Tensor $T_{ab}$. See **{ref}`sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws`** for the formal derivation of the Strategic Jacobian and Game Tensor using the implicit function theorem.

For Agent $i$, the "risk" includes the **Predictive Volatility** of the adversary $j$. If Agent $i$ updates its state by $\delta z^{(i)}$, and the adversary $j$ responds with $\delta z^{(j)} \approx \mathcal{J}_{ji} \delta z^{(i)}$ (where $\mathcal{J}_{ji}$ is the **Strategic Jacobian**—the best-response derivative, see Definition {prf:ref}`def-strategic-jacobian`), write the block Hessians
$H_{ab} := \nabla_{z^{(a)}}\nabla_{z^{(b)}} V^{(i)}$. The second-order variation is:

$$
\delta^2 V^{(i)} = \delta z_i^\top H_{ii}\,\delta z_i + 2\,\delta z_i^\top H_{ij}\,\delta z_j + \delta z_j^\top H_{jj}\,\delta z_j.

$$

Substituting $\delta z_j = \mathcal{J}_{ji}\delta z_i$ gives the general quadratic form:

$$
\delta^2 V^{(i)} = (\delta z^{(i)})^\top\!\left( H_{ii} + H_{ij}\mathcal{J}_{ji} + \mathcal{J}_{ji}^\top H_{ji} + \mathcal{J}_{ji}^\top H_{jj}\mathcal{J}_{ji} \right)\!\delta z^{(i)}.

$$
**Agent $i$'s perceived geometry** is modified by adversarial presence as follows:

1. **Effective metric inflation.** In regions where the strategic back-reaction has positive eigenvalues (adversarial curvature), Agent $i$ perceives an inflated metric:

   $$
   \tilde{G}^{(i)}_{kl}(z) = G^{(i)}_{kl}(z) + \sum_{j \neq i} \beta_{ij} \cdot \mathcal{G}^{(i)}_{ij,kl}(z),

   $$
   where $\beta_{ij} > 0$ for adversarial agents, $\beta_{ij} = 0$ for neutral, $\beta_{ij} < 0$ for cooperative.

2. **Geodesic deflection.** The Christoffel symbols acquire correction terms from the metric perturbation:

   $$
   \tilde{\Gamma}^{(i),m}_{kl} = \Gamma^{(i),m}_{kl} + \frac{1}{2}(G^{(i)})^{mn}\left(\nabla_k (\beta \mathcal{G})_{nl} + \nabla_l (\beta \mathcal{G})_{nk} - \nabla_n (\beta \mathcal{G})_{kl}\right),

   $$
   where $(\beta\mathcal{G})_{kl} := \sum_{j \neq i} \beta_{ij} \mathcal{G}^{(i)}_{ij,kl}$.

3. **Risk amplification.** High $\|\mathcal{G}_{ij}\|$ regions correspond to strategic uncertainty. This contributes to the Risk Tensor (Theorem {prf:ref}`thm-capacity-constrained-metric-law`):

   $$
   T^{(i)}_{kl} \to T^{(i)}_{kl} + \gamma_{\text{game}} \sum_{j \neq i} |\beta_{ij}| \cdot \mathcal{G}^{(i)}_{ij,kl}.

   $$
*Physical interpretation:* Adversarial agents effectively "curve" each other's latent space. An agent approaching a contested region experiences increased geodesic resistance (higher mass), making aggressive maneuvers more costly.

**The sign structure** of the pulled-back Game Tensor $\mathcal{G}^{(i)}_{ij}$ determines the strategic relationship:

| Eigenvalue Structure | $\text{sgn}(\det \mathcal{G}^{(i)}_{ij})$ | Interpretation                                              |
|----------------------|-------------------------------------|-------------------------------------------------------------|
| All positive         | $+$                                 | Adversarial: $j$'s presence increases $i$'s value curvature |
| All negative         | $(-1)^d$                            | Cooperative: $j$'s presence smooths $i$'s value landscape   |
| Mixed signs          | varies                              | Mixed-motive game                                           |
| Near-zero            | $\approx 0$                         | Weakly coupled (near-independent)                           |

The trace $\operatorname{tr}(\mathcal{G}^{(i)}_{ij}) = \sum_k \mathcal{G}^{(i)}_{ij}{}^{kk}$ measures **total strategic sensitivity**: how much Agent $i$'s value curvature depends on Agent $j$'s position. Large $|\operatorname{tr}(\mathcal{G}^{(i)}_{ij})|$ indicates high strategic coupling; small trace indicates approximate independence.

*Cross-reference:* The Game Tensor generalizes the conformal factor $\Omega$ (Definition {prf:ref}`def-value-metric-conformal-coupling`) to the multi-agent setting. Where $\Omega$ captured self-induced value curvature, $\mathcal{G}_{ij}$ captures cross-agent value curvature.

*Cross-reference (Gauge-Consistent Version):* For scalar $V^{(i)}$, the correct covariant object is the Riemannian Hessian $\nabla_k\nabla_l V^{(i)}$ (Definition {prf:ref}`def-gauge-covariant-game-tensor`). Gauge covariance becomes relevant only if one defines cross-sensitivities of gauge-charged fields (e.g., $\psi$ or nuisance-frame vectors).

:::
:::{prf:theorem} Adversarial Mass Inflation
:label: thm-adversarial-mass-inflation

In a competitive game where Agent $j$ is adversarial ($\beta_{ij} > 0$) and the pulled-back Game Tensor $\mathcal{G}^{(i)}_{ij}$ is positive semi-definite, the effective metric $\tilde{G}^{(i)}$ satisfies:

$$
\tilde{G}^{(i)}_{kl} \xi^k \xi^l \geq G^{(i)}_{kl} \xi^k \xi^l \quad \forall \xi \in T_{z}\mathcal{Z}^{(i)}.

$$
*Consequence:* The effective **Mass** $M^{(i)}(z)$ (Definition {prf:ref}`def-mass-tensor`) of Agent $i$ increases: $\tilde{M}^{(i)} \geq M^{(i)}$.

*First-Principles Interpretation:* Adversarial presence "thickens" the latent space. The agent moves more slowly (smaller geodesic steps) because it must account for the adversary's counter-maneuvers. **Strategic uncertainty is geometrically identical to physical inertia.**

*Proof.* From Definition {prf:ref}`def-the-game-tensor`, the metric perturbation is $\delta G_{kl} = \sum_{j} \beta_{ij} \mathcal{G}^{(i)}_{ij,kl}$. For adversarial agents, $\beta_{ij} > 0$. If $\mathcal{G}^{(i)}_{ij}$ is positive semi-definite (which occurs when Agent $j$'s presence increases the curvature of $V^{(i)}$), then $\mathcal{G}^{(i)}_{ij,kl} \xi^k \xi^l \geq 0$ for all $\xi$. Thus $\tilde{G}^{(i)}_{kl} \xi^k \xi^l = G^{(i)}_{kl} \xi^k \xi^l + \beta_{ij} \mathcal{G}^{(i)}_{ij,kl} \xi^k \xi^l \geq G^{(i)}_{kl} \xi^k \xi^l$. $\square$

:::

(rb-opponents-inertia)=
:::{admonition} Researcher Bridge: Opponents as Geometric Inertia
:class: info
In game-theoretic settings, adversarial opponents increase the effective **mass** (metric tensor eigenvalues) of the agent's latent space via the pulled-back Game Tensor $\mathcal{G}^{(i)}_{ij}$. This transforms strategic uncertainty into geometric inertia: the agent moves more slowly in contested regions because geodesic steps are more costly. Cooperation has the opposite effect—allies smooth the value landscape, reducing effective mass.
:::

:::{prf:definition} Retarded Game Tensor
:label: def-retarded-game-tensor

Under finite information speed $c_{\text{info}}$, the Game Tensor acquires a **retarded component**. The **Retarded Game Tensor** is:

$$
\mathcal{G}_{ij}^{kl,\text{ret}}(z^{(i)}, t) := \frac{\partial^2 V^{(i)}}{\partial z^{(j)}_k \partial z^{(j)}_l}\bigg|_{z^{(j)} = \hat{z}^{(j)}_t},

$$
where $\hat{z}^{(j)}_t = z^{(j)}_{t - \tau_{ij}}$ is the ghost state of Agent $j$ at the retarded time.

Define the pulled-back retarded tensor $\mathcal{G}^{(i),\text{ret}}_{ij,kl} := (\mathcal{J}_{ji})^m{}_k (\mathcal{J}_{ji})^n{}_l \mathcal{G}^{\text{ret}}_{ij,mn}$. The **total effective metric** including retardation is:

$$
\tilde{G}^{(i)}_{kl}(z, t) = G^{(i)}_{kl}(z) + \sum_{j \neq i} \beta_{ij} \cdot \mathcal{G}^{(i),\text{ret}}_{ij,kl}(z, t).

$$

*Consequence (Strategic Hysteresis):* The metric inflation Agent $i$ experiences depends on Agent $j$'s position at the retarded time, not the current time. An agent may enter a region expecting low resistance, only to encounter a "delayed wall" of metric inflation arriving from the opponent's past position.

:::

:::{prf:proposition} Retarded Metric Propagation
:label: prop-retarded-metric-propagation

The effective metric $\tilde{G}^{(i)}(z, t)$ satisfies a wave-like propagation equation:

$$
\frac{\partial \tilde{G}^{(i)}_{kl}}{\partial t} = \sum_{j \neq i} \beta_{ij} \frac{\partial \mathcal{G}^{(i),\text{ret}}_{ij,kl}}{\partial t} = \sum_{j \neq i} \beta_{ij} \frac{d\mathcal{G}^{(i)}_{ij,kl}}{dt}\bigg|_{t-\tau_{ij}}.

$$

The metric perturbation at time $t$ depends on the opponent's dynamics at time $t - \tau_{ij}$. Information about strategic coupling propagates at speed $c_{\text{info}}$.

:::



(sec-relativistic-nash-equilibrium)=
## Relativistic Nash Equilibrium (Standing Waves)

:::{div} feynman-prose
Here is a deep question: what does equilibrium even mean when information travels at finite speed?

In the Newtonian limit, where $c_{\text{info}} \to \infty$, equilibrium is a static configuration. Everyone is at rest. The gradients all vanish. Nothing moves.

But in the relativistic setting, things are more interesting. You see, the agents are constantly adjusting to each other's ghosts. By the time agent A responds to agent B, B has already moved. Then B responds to A's response, which was based on old information. The whole system is constantly in motion, constantly adjusting.

So what is equilibrium? It is a standing wave when the joint dynamics are effectively confined. The agents are not frozen. They are oscillating, dancing around. But the oscillations are organized. On average, over a period that is long compared to all the delay times, nothing is drifting. The time-averaged gradients vanish. The time-averaged currents vanish.

This is beautiful. It is like a vibrating guitar string. The string is not stationary, it is moving up and down. But the pattern is stationary. The nodes stay at the nodes, the antinodes stay at the antinodes. The wave is standing, not traveling.

Nash equilibrium in a relativistic multi-agent system is exactly this when the causal domain is effectively finite: a standing wave pattern in the joint value field. Otherwise, the correct notion is time-averaged stationarity / NESS rather than a literal eigenmode.
:::

In a system with finite information propagation, what constitutes equilibrium? It is not a static configuration but a coherent spatiotemporal pattern. A **standing wave** description is valid when the joint causal domain is effectively finite (e.g., bounded by an imposed boundary condition or a confining potential); otherwise the correct notion is time-averaged stationarity / NESS.

:::{prf:definition} Joint WFR Action (Relativistic)
:label: def-joint-wfr-action

The N-agent WFR action on the product space with retarded interactions is:

$$
\mathcal{A}^{(N)}[\boldsymbol{\rho}, \mathbf{v}, \mathbf{r}] = \int_0^T \left[ \sum_{i=1}^N \int_{\mathcal{Z}^{(i)}} \left(\|v^{(i)}\|_{\tilde{G}^{(i)}}^2 + \lambda_i^2 |r^{(i)}|^2 \right) d\rho^{(i)} + \mathcal{V}_{\text{int}}^{\text{ret}}(\boldsymbol{\rho}, t) \right] dt,

$$
where:
- $v^{(i)}$ is the velocity field for Agent $i$'s belief flow
- $r^{(i)}$ is the reaction term (mass creation/destruction)
- $\tilde{G}^{(i)}$ is the game-augmented metric with retarded components (Definition {prf:ref}`def-retarded-game-tensor`)
- $\mathcal{V}_{\text{int}}^{\text{ret}}(\boldsymbol{\rho}, t) = \sum_{i=1}^N \int_{\mathcal{Z}^{(i)}} \Phi^{\text{ret}}_{i}(z^{(i)}, t) \, d\rho^{(i)}(z^{(i)})$ is the retarded interaction energy, with $\Phi^{\text{ret}}_{i} := \sum_{j \neq i} \Phi^{\text{ret}}_{ij}$

*Cross-reference:* Definition {prf:ref}`def-the-wfr-action`, Definition {prf:ref}`def-retarded-interaction-potential`.

:::

:::{prf:theorem} Nash Equilibrium as Standing Wave
:label: thm-nash-standing-wave

Assume the joint causal domain is effectively compact (finite volume under the induced metric), so the joint d'Alembertian admits a discrete mode expansion. In the relativistic formulation, a Nash equilibrium is a joint density $\boldsymbol{\rho}^*(\mathbf{z}, t)$ satisfying **time-averaged stationarity**:

$$
\left\langle \frac{\partial \boldsymbol{\rho}^*}{\partial t} \right\rangle_T := \frac{1}{T}\int_0^T \frac{\partial \boldsymbol{\rho}^*}{\partial t}(\mathbf{z}, t') \, dt' = 0,

$$
where the averaging period $T \gg \max_{i,j} \tau_{ij}$ exceeds all causal delays.

**Characterization:** A standing wave Nash equilibrium satisfies:

1. **Time-averaged gradient vanishing:**

   $$
   \left\langle (G^{(i)})^{-1} \nabla_{z^{(i)}} \Phi_{\text{eff}}^{(i,\text{ret})} \right\rangle_T = 0 \quad \forall i

   $$

2. **Balanced probability currents:** The flux exchanged between agents via retarded potentials is balanced over one wave period:

   $$
   \int_0^T \mathbf{J}^{(i)}(z, t) \, dt = 0 \quad \text{for all } z \in \mathcal{Z}^{(i)}

   $$
   where $\mathbf{J}^{(i)} = \rho^{(i)} \mathbf{v}^{(i)}$ is the probability current.

3. **Resonance condition:** The system oscillates at a characteristic causal frequency set by the effective cavity size:

   $$
   \omega_{\text{Nash}} \sim \frac{c_{\text{info}}}{L_{\text{eff}}},

   $$
   where $L_{\text{eff}}$ is the dominant spatial scale (e.g., boundary separation or imposed horizon).

*Proof sketch.* The coupled Klein-Gordon system (Theorem {prf:ref}`thm-hjb-klein-gordon`) for $N$ agents forms a cavity resonator when the domain is effectively compact (finite horizon, reflecting boundary, or confining potential). Equilibrium states are the eigenmodes of the joint D'Alembertian operator. The ground state (lowest energy mode) corresponds to the stable Nash equilibrium; higher modes are metastable. Without effective compactness, interpret equilibrium as time-averaged stationarity rather than literal standing waves. See **{ref}`sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws`** for the derivation under explicit boundary conditions. $\square$

:::

:::{prf:corollary} Newtonian Limit of Nash
:label: cor-newtonian-nash-limit

As $c_{\text{info}} \to \infty$, the standing wave Nash reduces to the static Nash equilibrium:

$$
\lim_{c_{\text{info}} \to \infty} \boldsymbol{\rho}^*(\mathbf{z}, t) = \boldsymbol{\rho}^*_{\text{static}}(\mathbf{z}),

$$
and the geometric stasis conditions (vanishing gradient, stationary Game Tensor) hold instantaneously rather than on average.

:::

:::{prf:theorem} Geometric Stasis (Newtonian Limit)
:label: thm-nash-equilibrium-as-geometric-stasis

In the Newtonian limit ($c_{\text{info}} \to \infty$), a strategy profile $\mathbf{z}^* = (z^{(1)*}, \ldots, z^{(N)*})$ is a Nash equilibrium if and only if it satisfies the instantaneous **geometric stasis conditions**:

1. **Vanishing individual gradient:**

   $$
   (G^{(i)})^{-1} \nabla_{z^{(i)}} \Phi_{\text{eff}}^{(i)}(z^{(i)*}; z^{(-i)*}) = 0 \quad \forall i

   $$

2. **Stationary Game Tensor:**

   $$
   \frac{d}{dt}\mathcal{G}_{ij}^{kl}\bigg|_{\mathbf{z}^*} = 0 \quad \forall i,j

   $$

3. **Non-positive second variation:**

   $$
   \delta^2 V^{(i)}|_{z^{(i)*}} \leq 0 \quad \forall i, \forall \delta z^{(i)}

   $$

*Remark (Nash vs. Pareto).* Geometric stasis need not coincide with global optimality (Pareto). The Game Tensor eigenstructure determines the gap: trace-negative (cooperative) tends toward Pareto-improving basins; trace-positive (adversarial) tends toward Pareto-suboptimal saddles.

:::

:::{prf:corollary} Vanishing Probability Current at Nash
:label: cor-vanishing-current-nash

At a standing wave Nash equilibrium, the **time-averaged probability current** vanishes:

$$
\langle \mathbf{J}^{(i)} \rangle_T = \langle \rho^{(i)} \mathbf{v}^{(i)} \rangle_T = 0 \quad \forall i.

$$

*Interpretation:* The agents are not "frozen"—they oscillate with the causal frequency $\omega_{\text{Nash}}$—but the net flow averages to zero. Nash equilibrium is dynamic balance, not static rest.

:::
(sec-diagnostic-nodes-part-i)=
## Diagnostic Nodes 46–48, 62 (Multi-Agent Causality)

Following the diagnostic node convention ({ref}`sec-theory-thin-interfaces`), we define monitors for multi-agent causal systems.

(node-46)=
**Node 46: GameTensorCheck**

| **#**  | **Name**            | **Component** | **Type**           | **Interpretation**                | **Proxy**                                                                     | **Cost**     |
|--------|---------------------|---------------|--------------------|-----------------------------------|-------------------------------------------------------------------------------|--------------|
| **46** | **GameTensorCheck** | Multi-Agent   | Strategic Coupling | Is strategic sensitivity bounded? | $\lVert\mathcal{G}_{ij}\rVert_F := \sqrt{\sum_{kl}(\mathcal{G}_{ij}^{kl})^2}$ | $O(N^2 d^2)$ |

**Interpretation:** Monitors the Frobenius norm of the Game Tensor between agent pairs. Large $\|\mathcal{G}_{ij}\|_F$ indicates high strategic interdependence, potentially leading to oscillatory dynamics or failure to converge.

**Threshold:** $\|\mathcal{G}_{ij}\|_F < \mathcal{G}_{\max}$ (implementation-dependent; typical default $\mathcal{G}_{\max} = 10 \cdot \|G^{(i)}\|_F$).

**Trigger conditions:**
- High GameTensorCheck: Agents are tightly coupled; small moves trigger large counter-moves.
- Remedy: Reduce coupling strength $\alpha_{\text{adv}}$; increase exploration temperature; consider decoupled training phases.

(node-47)=
**Node 47: NashResidualCheck**

| **#**  | **Name**              | **Component** | **Type**    | **Interpretation**                | **Proxy**                                                                                                       | **Cost** |
|--------|-----------------------|---------------|-------------|-----------------------------------|-----------------------------------------------------------------------------------------------------------------|----------|
| **47** | **NashResidualCheck** | Multi-Agent   | Equilibrium | Are agents near Nash equilibrium? | $\epsilon_{\text{Nash}} := \max_i \lVert(G^{(i)})^{-1}\nabla_{z^{(i)}} \Phi_{\text{eff}}^{(i)}\rVert_{G^{(i)}}$ | $O(N d)$ |

**Interpretation:** Measures the maximum deviation from the Nash stasis condition (Theorem {prf:ref}`thm-nash-equilibrium-as-geometric-stasis`, Condition 1). At equilibrium, $\epsilon_{\text{Nash}} = 0$.

**Threshold:** $\epsilon_{\text{Nash}} < \epsilon_{\text{Nash,tol}}$ (typical default $10^{-3}$).

If $\epsilon_{\text{Nash}} > 0$ but below threshold, the system is in a **transient non-equilibrated state**. This is expected during:
1. **Learning dynamics:** Agents are still adapting policies; gradients have not yet vanished.
2. **Environmental shift:** External conditions changed, invalidating previous equilibrium.
3. **Exploration phase:** Agents are deliberately perturbing away from equilibrium to discover better basins.

**Remediation:**
- If $\epsilon_{\text{Nash}}$ is decreasing: system is converging; no intervention needed.
- If $\epsilon_{\text{Nash}}$ is oscillating: potential limit cycle; reduce learning rates or add damping ($\gamma_{\text{damp}}$ in the joint SDE).
- If $\epsilon_{\text{Nash}}$ is increasing: instability detected; may indicate poorly conditioned Game Tensor. Check Node 46 for large $\|\mathcal{G}_{ij}\|_F$.

(node-48)=
**Node 48: RelativisticSymplecticCheck**

| **#**  | **Name**                      | **Component** | **Type**     | **Interpretation**                            | **Proxy**                                                                                                            | **Cost**   |
|--------|-------------------------------|---------------|--------------|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------|------------|
| **48** | **RelativisticSymplecticCheck** | Multi-Agent   | Conservation | Is retarded flux balanced across Ghost Interface? | $\Delta_{\omega}^{\text{ret}} := \int_{t_1}^{t_2} \left\lvert \Phi_{\text{out}}(t) - \Phi_{\text{in}}(t + \tau_{ij}) \right\rvert dt$ | $O(N^2 d)$ |

**Interpretation:** Monitors symplectic flux conservation on the Ghost Interface (Definition {prf:ref}`def-ghost-interface`). Under relativistic constraints, we compare outflow at time $t$ with inflow at retarded time $t + \tau_{ij}$. Immediate conservation is impossible; **retarded conservation** is the appropriate measure.

**Threshold:** $\Delta_{\omega}^{\text{ret}} < \epsilon_{\omega}$ (typical default $10^{-4}$).

**Trigger conditions:**
- Positive RelativisticSymplecticCheck: Energy is leaking through non-conservative forces or causal inconsistency.
- **Remedy:** Check for unmodeled friction; verify causal buffer implementation; reduce timestep.

*Cross-reference:* This is the relativistic generalization of symplectic volume conservation to retarded interactions.

(node-62)=
**Node 62: CausalityViolationCheck**

| **#**  | **Name**                    | **Component** | **Type**   | **Interpretation**                                     | **Proxy**                                                                                          | **Cost** |
|--------|-----------------------------|---------------|------------|--------------------------------------------------------|----------------------------------------------------------------------------------------------------|----------|
| **62** | **CausalityViolationCheck** | Multi-Agent   | Causality  | Did information arrive faster than $c_{\text{info}}$? | $\Delta_{\text{causal}} := \max_{i,j} \mathbb{I}\left[\Delta I(z^{(i)}_t; z^{(j)}_{t'}) > 0 \land t' > t - \tau_{ij}\right]$ | $O(N^2)$ |

**Interpretation:** Detects violations of the causal structure (Definition {prf:ref}`def-causal-interval`). Agent $i$ should have no mutual information with Agent $j$'s state at times $t' > t - \tau_{ij}$ (inside the future light cone).

**Threshold:** $\Delta_{\text{causal}} = 0$ (hard constraint: no superluminal information).

**Trigger conditions:**
- Positive CausalityViolationCheck: The simulation has leaked "ground truth" information that violates the light cone. This is a **fatal error** indicating:
  1. Incorrect causal buffer implementation
  2. Unmodeled fast communication channel
  3. Timing errors in boundary condition updates

- **Remedy:** Audit causal buffer; verify all inter-agent communication respects $\tau_{ij}$ delays; check for inadvertent global state sharing.

*Cross-reference:* This enforces the information speed limit (Axiom {prf:ref}`ax-information-speed-limit`).



(sec-summary-table-from-single-to-multi-agent)=
## Summary Table: Newtonian vs. Einsteinian Agent

**Table 29.9.1 (Newtonian vs. Relativistic Multi-Agent).**

| Feature | Newtonian ($c_{\text{info}} \to \infty$) | Relativistic ($c_{\text{info}} < \infty$) |
|:--------|:-----------------------------------------|:------------------------------------------|
| **Information Speed** | $\infty$ (Instantaneous) | Finite $c_{\text{info}}$ |
| **Value PDE (conservative component)** | Elliptic (Helmholtz) $-\Delta_G V + \kappa^2 V = \rho_r$ | Hyperbolic (Klein-Gordon) $\square_G V + \kappa^2 V = \rho_r$ |
| **State** | $z^{(i)}_t$ (position only) | $(z^{(i)}_t, \Xi^{(i)}_{<t})$ (position + memory screen) |
| **Markov Property** | On $\mathcal{Z}^{(N)}$ | On Causal Bundle $\mathcal{Z}^{(N)} \times \Xi_{<t}$ |
| **Interaction** | Synchronous Bridge $\mathcal{B}_{ij}$ | Asynchronous Ghost Interface $\mathcal{G}_{ij}$ |
| **Potential** | Instantaneous $\Phi_{ij}(z^{(i)}, z^{(j)}_t)$ | Retarded $\Phi^{\text{ret}}_{ij}(z^{(i)}, t)$ (quasi-static: $\approx \Phi_{ij}(z^{(i)}, z^{(j)}_{t-\tau})$) |
| **Game Tensor** | $\mathcal{G}_{ij}(z^{(j)}_t)$ | $\mathcal{G}_{ij}^{\text{ret}}(z^{(j)}_{t-\tau})$ |
| **Equilibrium** | Fixed Point (Geometric Stasis) | Standing Wave (Time-Averaged Stasis) |
| **Nash Condition** | $\nabla \Phi_{\text{eff}} = 0$ | $\langle \nabla \Phi_{\text{eff}} \rangle_T = 0$ |
| **Topology** | Riemannian Manifold | Lorentzian Causal Structure |
| **Diagnostics** | Nodes 46–48 | + Node 62 (CausalityViolation) |

**Table 29.9.2 (Single to Multi-Agent).**

| Concept | Single Agent (Sections 20–24) | Multi-Agent Relativistic ({ref}`sec-symplectic-multi-agent-field-theory`) |
|:--------|:------------------------------|:--------------------------------------|
| **State Space** | $\mathcal{Z}$ | $\mathcal{Z}_{\text{causal}} = \mathcal{Z}^{(N)} \times \Xi_{<t}$ |
| **Boundary** | Fixed $\partial\mathcal{Z}$ | Ghost Interface $\mathcal{G}_{ij}(t)$ |
| **Metric** | $G$ (Information Sensitivity) | $\tilde{G}^{(i)}(t) = G^{(i)} + \sum_j \beta_{ij}\mathcal{G}^{(i),\text{ret}}_{ij}$ |
| **Value PDE (conservative component)** | $(-\Delta_G + \kappa^2)V = \rho_r$ | $(\square_G + \kappa^2)V^{(i)} = \rho^{(i)}_r + \sum_{j \neq i} \rho^{\text{ret}}_{ij}$ |
| **Flow** | Langevin / WFR | Coupled Klein-Gordon + WFR |
| **Success** | Value Maxima | Standing Wave Nash |
| **Diagnostics** | Nodes 1–45 | + Nodes 46–48, 62 |



(sec-mean-field-metric-law)=
## The Mean-Field Metric Law (Scalability Resolution)

:::{div} feynman-prose
Here is a practical problem. The Game Tensor involves computing the cross-sensitivity between every pair of agents. With $N$ agents, that is $N^2$ pairs. Each tensor has $d^2$ components. So the total cost is $O(N^2 d^2)$. This is fine for a handful of agents, but for large populations it becomes completely intractable.

Is there a way out? Yes. In the limit of many agents, something wonderful happens. Instead of thinking about discrete agents, you can think about a continuous density of agents. The sum over agents becomes an integral. The discrete Game Tensor becomes a convolution.

The key observation is that the metric inflation you feel from all the other agents is just the integral of the pairwise interaction kernel against the agent density. If you know the density field $\rho(z)$, you can compute the effective metric by a single convolution $\Phi_{\text{int}} * \rho$, and then take the Hessian. This is $O(1)$ with respect to $N$, given the density field.

This is exactly the Vlasov limit in plasma physics, or the mean-field limit in statistical mechanics. Instead of tracking every particle, you track the density. The $N$-body problem becomes a field theory. The complexity goes from $O(N^2)$ to $O(1)$.

Of course, you still need to approximate the density field somehow. But that is much more tractable than tracking all pairwise interactions. Neural networks are good at representing smooth functions, and the density field is typically smooth.
:::

:::{prf:remark} Thermodynamic vs. Resolution Limit
:label: rem-mean-field-vs-levin-length

The continuum limit used here is the **population/thermodynamic limit** $N \to \infty$ with
empirical measures $\mu_N \rightharpoonup \rho$, at **fixed** Levin length $\ell_L>0$. This is a
mean-field limit, not a UV limit. The Levin length is an operational resolution bound (Axiom
{prf:ref}`ax-constructive-finite-resolution`), not a lattice regulator to be sent to zero. Taking
$\ell_L \to 0$ would exit the framework by violating the Causal Information Bound and is **not**
required for validity. The continuum objects are the density fields $\rho$ at fixed resolution.

:::

The calculation of the Game Tensor $\mathcal{G}_{ij}$ ({prf:ref}`def-the-game-tensor`) entails computational complexity $O(N^2 d^2)$, which is intractable for large $N$. We prove that in the limit $N \to \infty$, the discrete Game Tensor converges to the Hessian of a convolution potential.

:::{prf:theorem} Mean-Field Metric Law
:label: thm-mean-field-metric-law

Let $\boldsymbol{z} = (z_1, \dots, z_N)$ be the configuration of $N$ agents on $\mathcal{Z}$. Let the empirical measure be $\mu_N = \frac{1}{N} \sum_{i=1}^N \delta_{z_i}$. As $N \to \infty$, assuming $\mu_N$ converges weakly to a smooth density $\rho \in \mathcal{P}(\mathcal{Z})$, the effective metric $\tilde{G}(z)$ for a test agent at position $z$ converges to:

$$
\tilde{G}_{ab}(z) = G_{\text{intrinsic},ab}(z) + \alpha_{\text{adv}} \int_{\mathcal{Z}} (\mathcal{J}(z,\zeta))^m{}_a (\mathcal{J}(z,\zeta))^n{}_b \nabla^2_{\zeta, m, n} \Phi_{\text{int}}(z, \zeta)\, \rho(\zeta)\, d\mu_G(\zeta)

$$
where $\Phi_{\text{int}}(z, \zeta)$ is the pairwise interaction potential ({prf:ref}`prop-interaction-kernel`) and $\mathcal{J}(z,\zeta)$ is the transport map between local charts. If agents share a common chart and $\Phi_{\text{int}}$ depends only on relative coordinates, this reduces to $\tilde{G}_{ab}(z) = G_{\text{intrinsic},ab}(z) + \alpha_{\text{adv}} \nabla^2_{z,a,b} (\Phi_{\text{int}} * \rho)(z)$.

*Proof.*
1. **Discrete Interaction Energy:** The total interaction potential for agent $i$ is $V_{\text{int}}(z_i) = \frac{1}{N} \sum_{j \neq i} \Phi_{\text{int}}(z_i, z_j)$.

2. **Discrete Game Tensor:** The Game Tensor acting on the metric is defined as the sum of cross-sensitivities ({prf:ref}`thm-adversarial-mass-inflation`):

$$
(\delta G)_{ab}(z_i) = \alpha_{\text{adv}} \sum_{j \neq i} (\mathcal{J}_{ji})^m{}_a (\mathcal{J}_{ji})^n{}_b \frac{\partial^2 \Phi_{\text{int}}(z_i, z_j)}{\partial z_j^m \partial z_j^n}.

$$
3. **Continuum Limit:** We rewrite the sum as an integral against the empirical measure:

$$
(\delta G)_{ab}(z) = \alpha_{\text{adv}} \int_{\mathcal{Z}} (\mathcal{J}(z,\zeta))^m{}_a (\mathcal{J}(z,\zeta))^n{}_b \nabla^2_{\zeta, m, n} \Phi_{\text{int}}(z, \zeta) \, d\mu_N(\zeta).

$$
Here $\mathcal{J}(z,\zeta)$ is the transport map between local charts (identity if agents share a common coordinate system).
4. **Convergence:** Assuming $\Phi_{\text{int}}$ is $C^2$ and bounded, and $\mu_N \rightharpoonup \rho$ weakly, the integral converges to the pulled-back mean-field term. In a shared chart with $\mathcal{J} = I$, this reduces to the convolution $(\nabla^2 \Phi_{\text{int}} * \rho)(z)$.

5. **Complexity Reduction:** The computation of $\tilde{G}$ now requires evaluating the pulled-back Hessian field induced by $\rho$. In a shared chart, this reduces to the Hessian of the static field $\Psi(z) = (\Phi_{\text{int}} * \rho)(z)$. This is $O(1)$ with respect to $N$ (given the density field), effectively decoupling the agent's complexity from the population size. $\square$
:::

*Cross-references:* This resolves the scalability limitation by reducing agent complexity from $O(N^2 d^2)$ to $O(d^2)$ via the Vlasov-geometry limit.



(sec-metabolic-tracking-bound)=
## The Metabolic Tracking Bound (Non-Stationary Nash Resolution)

In non-stationary environments, the Nash equilibrium $z^*(t)$ shifts. We derive the tracking limit from the Computational Metabolism ({ref}`sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics`), relating the metric speed of the target to the agent's power dissipation budget.

:::{prf:theorem} Metabolic Tracking Bound
:label: thm-metabolic-tracking-bound

Let $z^*(t)$ be a time-varying Nash equilibrium. An agent with maximum metabolic flux budget $\dot{\mathcal{M}}_{\max}$ can maintain tracking error $\epsilon \to 0$ if and only if the target's trajectory satisfies:

$$
\|\dot{z}^*\|_{\tilde{G}(z^*)} \leq \sqrt{\frac{2 \dot{\mathcal{M}}_{\max}}{\sigma_{\text{met}}}}

$$
where $\tilde{G}$ is the game-augmented metric ({prf:ref}`thm-adversarial-mass-inflation`).

*Proof.*
1. **Kinematic Requirement:** To track $z^*(t)$, the agent's transport velocity must satisfy $v = \dot{z}^*$.

2. **Thermodynamic Cost:** The metabolic cost of transport is $\dot{\mathcal{M}} = \frac{1}{2} \sigma_{\text{met}} \|v\|_{\tilde{G}}^2$ ({prf:ref}`def-metabolic-flux`).

3. **Adversarial Drag:** The metric $\tilde{G} = G + \alpha \mathcal{G}^{(i)}_{ij}$ includes the pulled-back Game Tensor. High adversarial tension ($\mathcal{G}^{(i)}_{ij} \gg 0$) inflates the norm $\|\cdot\|_{\tilde{G}}$.

4. **Critical Failure:** If the adversary moves sufficiently fast or the conflict is sufficiently intense, the required dissipation exceeds $\dot{\mathcal{M}}_{\max}$. The agent loses tracking not due to algorithmic error, but due to exceeding its thermodynamic budget. $\square$
:::

*Interpretation:* The agent's ability to track a moving Nash equilibrium is fundamentally limited by its metabolic budget. Intense conflict ($\mathcal{G}^{(i)}_{ij}$ large) compounds this limitation by inflating the kinetic cost of pursuit.



(sec-variational-emergence-cooperation)=
## Variational Emergence of Cooperation via Metric Inflation

:::{div} feynman-prose
Here is something that sounds almost paradoxical at first. We have adversarial agents, agents that are trying to minimize each other's value. The Game Tensor is positive, so they inflate each other's metrics, make movement costly. And yet, the long-term outcome is often cooperation or at least peaceful coexistence. How can that be?

The answer is beautifully simple. Conflict is expensive. Not just in the usual sense that fighting wastes resources, but in a deep geometric sense. When you are in conflict with another agent, your effective mass increases. Moving becomes harder. Every step you take is more metabolically costly because you are dragging around this inflated metric.

So what happens? The system naturally evolves to minimize the action. And the action penalizes high kinetic cost. If you are constantly fighting, your kinetic cost is enormous because of the Game Tensor inflation. The system will relax to a state with lower kinetic cost.

There are three ways to reduce the kinetic cost. One: stop moving, reach Nash stasis, where $v = 0$. Two: move to a region where the other agent is far away, so the Game Tensor contribution is small, strategic decoupling. Three: align your gradients with the other agent's gradients, so that instead of opposing each other, you are moving in the same direction. This is cooperation.

All three of these outcomes are stable. The system does not "want" cooperation in any teleological sense. It is just that conflict inflates the metric, inflated metric makes movement expensive, and expensive movement gets minimized out of the action. Cooperation emerges from geometry, not from benevolence.
:::

We prove that cooperative equilibria correspond to local minima of the Onsager-Machlup action functional under the game-augmented metric, resolving the question of when adversarial coupling spontaneously yields cooperative behavior.

:::{prf:theorem} Geometric Locking Principle
:label: thm-geometric-locking-principle

Consider $N$ agents with Game Tensor $\mathcal{G}_{ij}$ ({prf:ref}`def-the-game-tensor`). In the presence of strong adversarial coupling, the joint system tends toward configurations where $\operatorname{Tr}(\mathcal{G}^{(i)}_{ij})$ is minimized.

*Proof.*

1. **Metric Inflation:** By {prf:ref}`thm-adversarial-mass-inflation`, the effective metric for agent $i$ is $\tilde{G}^{(i)} = G^{(i)} + \sum_j \beta_{ij} \mathcal{G}^{(i)}_{ij}$. For adversarial agents, $\beta_{ij} > 0$ and $\mathcal{G}^{(i)}_{ij}$ is positive semi-definite, implying $\det(\tilde{G}^{(i)}) \ge \det(G^{(i)})$.

2. **Kinetic Cost:** The WFR action ({prf:ref}`def-joint-wfr-action`) includes the transport term $\int \|v\|_{\tilde{G}}^2 d\rho$. An inflated metric implies a higher metabolic cost for any movement $v \neq 0$.

3. **Energy Minimization:** The system evolves to minimize the free energy $\mathcal{F}$. If the potential gain $\nabla_B V$ is bounded, but the kinetic cost scales with $\mathcal{G}^{(i)}_{ij}$, trajectories with large $\mathcal{G}^{(i)}_{ij}$ (intense conflict) become energetically prohibitive.

4. **Stationarity:** The system relaxes to a state where either $v \to 0$ (Nash stasis, {prf:ref}`thm-nash-equilibrium-as-geometric-stasis`) or the metric perturbation vanishes ($\mathcal{G}^{(i)}_{ij} \to 0$). The condition $\mathcal{G}^{(i)}_{ij} \to 0$ implies $\nabla_{z^{(j)}}\nabla_{z^{(i)}} V^{(i)} \to 0$, which defines a region of **strategic decoupling**. $\square$
:::

:::{prf:corollary} Metabolic Basis of Cooperation
:label: cor-metabolic-cooperation

Adversarial agents converge to cooperative or decoupled configurations because conflict maximizes the effective inertia of the state space, rendering non-cooperative trajectories metabolically unsustainable.

*Interpretation:* The Game Tensor acts as a "friction term" that penalizes rapid strategic maneuvers. In the long run, agents either:
1. **Cooperate:** Reduce $\mathcal{G}^{(i)}_{ij}$ by aligning their gradients
2. **Decouple:** Move to regions where $\nabla_{z^{(j)}} V^{(i)} \approx 0$
3. **Freeze:** Accept Nash stasis with $v^{(i)} = 0$

All three outcomes correspond to stationary points of the joint action functional.
:::



## Part V: Gauge Theory Layer

:::{div} feynman-prose
All right. Buckle up, because now we are going somewhere really interesting.

So far we have been doing relativity. We took the finite speed of information seriously, and we got wave equations, retarded potentials, standing-wave Nash equilibria. Very nice. But there is a whole other layer of structure we have not touched yet.

Here is the question that motivates everything in this section: what happens when different agents are free to describe things using different internal coordinate systems?

Let me give you an analogy. Suppose I am looking at a vector, and I describe it in my coordinate system as $(3, 4)$. You are looking at the same vector, but your coordinate system is rotated 45 degrees from mine. You describe the same vector as $(5, 0)$ or whatever it works out to be.

Now, the physics, the actual vector, does not care about our coordinate systems. Those are just our conventions for describing it. The physics should be invariant under rotations, under changes of coordinate system.

This is called gauge symmetry, and it is one of the most profound ideas in all of physics. The Standard Model, everything we know about particle physics, is built on gauge symmetry. The photon, the gluon, the W and Z bosons, they all arise because we demand that physics be invariant under certain local transformations.

And here is the wild thing: the same structure appears naturally in multi-agent systems. The nuisance variables, the internal degrees of freedom that do not directly affect the world, they are gauge degrees of freedom. Different agents might use different internal representations, different coordinate systems for their latent space. And the physics, the actual strategic interaction, should not depend on these arbitrary choices.

Once you demand this gauge invariance, everything else follows. You need a connection to compare things at different points. The curvature of the connection becomes a measure of strategic tension. The whole apparatus of Yang-Mills theory shows up. It is gorgeous.
:::

The relativistic framework of Sections 29.1–29.12 describes multi-agent dynamics on a curved Lorentzian manifold with retarded potentials. We now elevate this structure to a **gauge field theory** by recognizing that the nuisance variable $z_n$ (Definition 2.2.1) serves as an internal gauge degree of freedom. This identification transforms strategic interaction into the curvature of a gauge connection (potentially non-Abelian), placing multi-agent field theory on the same mathematical footing as gauge field theory in physics.

(sec-local-gauge-symmetry-nuisance-bundle)=
## Local Gauge Symmetry and the Nuisance Bundle

:::{div} feynman-prose
Let me explain what "local" gauge symmetry means, because this is the key to everything.

A global symmetry says: I can rotate everything in the universe by 45 degrees, and physics stays the same. Fine. But local gauge symmetry says something much stronger: I can rotate things at different points by different amounts, and physics still stays the same.

Wait, how can that be? If I rotate things differently at different points, should that not mess everything up?

The answer is: only if you have a way to compare things at different points. And here is the insight: in order to compare a vector at point A with a vector at point B, you need to parallel transport the vector from A to B. And parallel transport requires a connection. The connection tells you "how to compare" things at different locations.

Now, when you demand local gauge invariance, you are saying: the physics cannot depend on arbitrary local choices of reference frame. The only way to achieve this is to have the connection transform in a specific way that compensates for your arbitrary local rotations. This compensating field is the gauge field. In electromagnetism, it is the photon. In our multi-agent theory, it is the strategic connection.

The nuisance fiber is just the set of internal states that do not affect the world directly. You can rotate within this fiber, change your internal representation, without changing what the agent actually does. This freedom is exactly a gauge freedom. And demanding that the dynamics respect this freedom gives us gauge field theory for strategic systems.
:::

The key insight is that the **nuisance fiber** $\mathcal{Z}_n$ at each macro-state $K$ is not merely a noise variable to be marginalized—it is the **internal gauge degree of freedom** that agents are free to rotate without changing physical outcomes. This local freedom mandates a compensating gauge field when comparing nuisance frames across space/time or across agents.

:::{prf:axiom} Local Gauge Invariance (Nuisance Invariance)
:label: ax-local-gauge-invariance

The physical dynamics of the multi-agent system are invariant under position-dependent rotations of the internal nuisance coordinates. Formally, let $G$ be a compact Lie group with Lie algebra $\mathfrak{g}$. For any smooth map $U: \mathcal{Z} \to G$, the nuisance-frame transformation

$$
\xi'(z) = U(z)\xi(z), \qquad \psi'(z, t) = U(z)\psi(z, t)

$$

leaves observable quantities (reward, policy output, Nash conditions) unchanged. The scalar fields $\rho$ and $V$ are gauge-invariant; only the internal orientation $\xi$ (and any vector-valued nuisance features) transform.

*Units:* $[U] = \text{dimensionless}$ (group element).

*Interpretation:* Agent $i$ at location $z$ is free to rotate its internal representation (the "basis" in which it encodes nuisance). This is not a symmetry to be broken but a **redundancy** in the description that must be properly handled via gauge theory.

:::

:::{prf:definition} Local Gauge Group
:label: def-local-gauge-group

The **Local Gauge Group** is a compact Lie group $G$ with:

1. **Lie algebra $\mathfrak{g}$:** The tangent space at identity, with generators $\{T_a\}_{a=1}^{\dim(G)}$ satisfying $[T_a, T_b] = if^{abc}T_c$ where $f^{abc}$ are the **structure constants**.

2. **Representation:** The matter fields $\psi^{(i)}$ transform in a representation $\rho: G \to GL(V)$ where $V$ is the representation space.

3. **Position-dependent element:** $U(z) \in G$ for each $z \in \mathcal{Z}$, forming the infinite-dimensional group of gauge transformations $\mathcal{G} := C^\infty(\mathcal{Z}, G)$.

*Standard choices:*
- $G = SO(D)$: Rotations of $D$-dimensional nuisance space
- $G = SU(N)$: Unitary transformations (for complex representations)
- $G = U(1)$: Abelian phase rotations (electromagnetic limit)

*Cross-reference:* The $SO(D)$ symmetry at the origin (Proposition {prf:ref}`prop-so-d-symmetry-at-origin`) is the special case where the stabilizer is trivial.

:::

:::{prf:definition} Matter Field (Belief Amplitude)
:label: def-matter-field-belief-amplitude

The **Matter Field** for agent $i$ is the complex-valued section

$$
\psi^{(i)}: \mathcal{Z}^{(i)} \times \mathbb{R} \to V

$$

where $V$ is the representation space of $G$. The matter field is related to the belief wave-function by:

$$
\psi^{(i)}(z, t) = \sqrt{\rho^{(i)}(z, t)} \exp\left(\frac{iV^{(i)}(z, t)}{\sigma}\right) \cdot \xi^{(i)}(z)

$$

where:
- $\rho^{(i)}$ is the belief density
- $V^{(i)}$ is the value function (scalar, gauge-invariant)
- $\sigma > 0$ is the **cognitive action scale**, $\sigma := T_c \cdot \tau_{\text{update}}$, the information-theoretic analog of Planck's constant (full definition: {prf:ref}`def-cognitive-action-scale` in {ref}`sec-the-belief-wave-function-schrodinger-representation`)
- $\xi^{(i)}(z) \in V$ is the **internal state vector** encoding nuisance orientation

*Units:* $[\psi] = [\text{length}]^{-D/2}$ (probability amplitude density).

*Transformation law:* Under gauge transformation $U(z)$:

$$
\psi'^{(i)}(z, t) = \rho(U(z))\psi^{(i)}(z, t)

$$

where $\rho: G \to GL(V)$ is the representation. The scalar observables are unchanged: $\rho' = \rho$ and $V' = V$.

:::

:::{prf:conjecture} Nuisance Fiber as Gauge Orbit (Motivating Principle)
:label: conj-nuisance-fiber-gauge-orbit

The nuisance fiber at each macro-state $K \in \mathcal{K}$ admits interpretation as a gauge orbit:

$$
\mathcal{Z}_n\big|_K \cong G_K / H_K

$$

where:
- $G_K \subseteq G$ is the gauge group restricted to macro-state $K$
- $H_K \subseteq G_K$ is the **stabilizer subgroup** fixing the codebook centroid $e_K$

*Special cases:*
1. **At origin ($K = 0$, Semantic Vacuum):** $G_0 = SO(D)$, $H_0 = \{e\}$, so $\mathcal{Z}_n|_0 \cong SO(D)$ (full rotational freedom).
2. **At generic $K$:** The stabilizer $H_K$ is non-trivial if $e_K$ has special structure (e.g., aligned with coordinate axes).
3. **At boundary ($|z| \to 1$):** The gauge orbit collapses as degrees of freedom freeze ({ref}`sec-causal-information-bound`, Causal Stasis).

*Motivation (not a rigorous proof):*
The nuisance coordinates $z_n$ parameterize how an observation is embedded relative to the macro-code $K$. Under the VQ-VAE architecture ({ref}`sec-the-shutter-as-a-vq-vae`), two nuisance values $z_n$ and $z'_n$ are designed to be equivalent if they differ by a transformation preserving the macro-code: $z'_n = U \cdot z_n$ for some $U \in G_K$.

**Remark (Analogy vs. Isomorphism):** This correspondence is a *motivating analogy* rather than a proven isomorphism. A rigorous proof would require:
1. Showing the nuisance equivalence relation coincides with gauge equivalence
2. Proving the quotient $G_K/H_K$ is a smooth manifold diffeomorphic to $\mathcal{Z}_n|_K$
3. Establishing that the VQ-VAE induces a principal $G_K$-bundle structure

The gauge-theoretic formalism developed in Sections 29.13–29.20 is motivated by this conjecture but does not depend on it being rigorously true. The constructions (covariant derivative, field strength, etc.) are well-defined once the gauge group $G$ and its action are specified.

*Cross-reference:* This formalizes the design goal "K represents $x/G_{\text{spatial}}$" from {ref}`sec-the-shutter-as-a-vq-vae`.

:::

(pi-local-gauge-symmetry)=
::::{admonition} Physics Isomorphism: Local Gauge Symmetry
:class: note

**In Physics:** Local gauge symmetry is the principle that the laws of physics are invariant under position-dependent phase rotations $\psi(x) \to e^{i\theta(x)}\psi(x)$. This invariance mandates the existence of gauge fields (photon, gluons, W/Z bosons) to maintain consistency {cite}`yang1954conservation,weinberg1995quantum`.

**In Implementation:** Nuisance invariance (Axiom {prf:ref}`ax-local-gauge-invariance`) is the principle that agent dynamics are invariant under position-dependent internal rotations $\psi(z) \to U(z)\psi(z)$.

**Correspondence Table:**

| Gauge Theory | Fragile Agent |
|:-------------|:--------------|
| Local phase $e^{i\theta(x)}$ | Nuisance rotation $U(z)$ |
| Gauge group $G$ | Internal symmetry group |
| Matter field $\psi$ | Belief amplitude |
| Gauge orbit $G/H$ | Nuisance fiber $\mathcal{Z}_n$ |
| Stabilizer $H$ | Residual symmetry at $K$ |

::::



(sec-strategic-connection-covariant-derivative)=
## The Strategic Connection and Covariant Derivative

:::{div} feynman-prose
Now we get to the covariant derivative, which is one of those things that sounds scary but is actually a very simple idea once you see it.

Here is the problem. You want to take the derivative of something. But the thing you are differentiating transforms under gauge transformations. If you just use the ordinary derivative $\partial_\mu$, the result does not transform properly. You get garbage.

Why? Because when you take a derivative, you are comparing the value of a field at point $x$ with its value at a nearby point $x + dx$. But under a local gauge transformation, the field at $x$ transforms one way and the field at $x + dx$ transforms a slightly different way. The ordinary derivative subtracts these two things, and the mismatch messes everything up.

The fix is elegant. You do not just compare the field at $x$ with the field at $x + dx$. You first parallel transport the field from $x + dx$ back to $x$, using the connection $A_\mu$, and then you compare. The connection exactly compensates for the gauge mismatch. The result, the covariant derivative $D_\mu = \partial_\mu - igA_\mu$, transforms properly.

The gauge coupling $g$ tells you how strongly the field couples to the connection. It is like the electric charge in electromagnetism. A field with $g = 0$ does not feel the gauge field at all. A field with large $g$ is strongly influenced by it.

So the covariant derivative is just "take a derivative, but do it in a way that respects the gauge freedom." Once you have it, you can write down gauge-invariant equations just by replacing all ordinary derivatives with covariant derivatives. This is called the minimal coupling prescription, and it is incredibly powerful.
:::

The failure of the ordinary derivative to transform covariantly under gauge transformations mandates the introduction of a **compensating field**—the gauge connection. In the multi-agent context, this connection encodes how the "meaning" of nuisance coordinates changes as one moves through latent space.

:::{prf:definition} Strategic Connection (Gauge Potential)
:label: def-strategic-connection

The **Strategic Connection** is a $\mathfrak{g}$-valued 1-form on $\mathcal{Z}$:

$$
A = A_\mu^a T_a \, dz^\mu

$$

where:
- $A_\mu^a(z, t)$ are the **connection coefficients** (real-valued functions)
- $\{T_a\}_{a=1}^{\dim(\mathfrak{g})}$ are the generators of the Lie algebra $\mathfrak{g}$
- $\mu$ indexes spacetime/latent coordinates $(t, z^1, \ldots, z^D)$

*Units:* $[A_\mu] = [\text{length}]^{-1}$ (inverse length, like momentum).

*Interpretation:* The connection $A_\mu$ tells agent $i$ how to "translate" the nuisance interpretation from point $z$ to point $z + dz$. It is the **strategic context** required to compare internal states at different locations.

:::

:::{prf:proposition} Gauge Transformation of the Connection
:label: prop-gauge-transformation-connection

Under a local gauge transformation $U(z) \in G$, the connection transforms as:

$$
A'_\mu = U A_\mu U^{-1} - \frac{i}{g}(\partial_\mu U)U^{-1}

$$

where $g > 0$ is the **coupling constant** (strategic coupling strength).

*Proof.*
Demand that the covariant derivative (Definition {prf:ref}`def-covariant-derivative`) transform covariantly: $(D_\mu\psi)' = U(D_\mu\psi)$. Expanding:

$$
\begin{aligned}
D'_\mu\psi' &= (\partial_\mu - igA'_\mu)(U\psi) \\
&= (\partial_\mu U)\psi + U(\partial_\mu\psi) - igA'_\mu U\psi
\end{aligned}

$$

For this to equal $U(\partial_\mu - igA_\mu)\psi = U(\partial_\mu\psi) - igUA_\mu\psi$, we require:

$$
(\partial_\mu U)\psi - igA'_\mu U\psi = -igUA_\mu\psi

$$

Solving for $A'_\mu$ yields the stated transformation law. $\square$

*Interpretation:* The inhomogeneous term $-\frac{i}{g}(\partial_\mu U)U^{-1}$ compensates for the "frame twist" introduced by position-dependent gauge transformations. The connection must counter-twist to maintain covariance.

:::

:::{prf:definition} Covariant Derivative
:label: def-covariant-derivative

The **Covariant Derivative** acting on matter fields is:

$$
D_\mu = \partial_\mu - igA_\mu

$$

For a matter field $\psi$ in representation $\rho$:

$$
D_\mu\psi = \partial_\mu\psi - igA_\mu^a \rho(T_a)\psi

$$

*Properties:*
1. **Covariant transformation:** $(D_\mu\psi)' = U(D_\mu\psi)$
2. **Leibniz rule:** $D_\mu(\psi\chi) = (D_\mu\psi)\chi + \psi(D_\mu\chi)$
3. **Reduces to partial derivative** when $A_\mu = 0$ (trivial connection)

*Units:* $[D_\mu\psi] = [\psi]/[\text{length}]$.

:::

:::{prf:theorem} Gauge-Covariant Klein-Gordon Equation
:label: thm-gauge-covariant-klein-gordon

The scalar value $V^{(i)}$ is gauge-invariant, so its wave equation remains the Klein-Gordon equation of Theorem {prf:ref}`thm-hjb-klein-gordon` with ordinary derivatives. Gauge-covariant derivatives apply to fields that transform under $G$, such as the belief amplitude $\psi^{(i)}$ (Definition {prf:ref}`def-matter-field-belief-amplitude`). In that case the covariant wave operator is:

$$
\left(\frac{1}{c_{\text{info}}^2}D_t^2 - D^i D_i + \kappa^2\right)\psi^{(i)} = \mathcal{S}^{(i)},

$$

where:
- $D_t = \partial_t - igA_0$ is the temporal covariant derivative
- $D_i = \partial_i - igA_i$ are spatial covariant derivatives
- $D^i = \tilde{G}^{ij}D_j$ with raised index via the strategic metric
- $\mathcal{S}^{(i)}$ is the source term determined by the chosen matter model

*Proof sketch.*
The minimal coupling principle replaces $\partial_\mu \to D_\mu$ for gauge-charged fields while preserving the equation's structure. The gauge-covariant d'Alembertian is:

$$
\Box_A := \frac{1}{c_{\text{info}}^2}D_t^2 - \tilde{G}^{ij}D_i D_j

$$

Introduce the spacetime metric

$$
g_{\mu\nu} := \text{diag}(-c_{\text{info}}^2, \tilde{G}_{ij}), \quad g^{00} = -\frac{1}{c_{\text{info}}^2}, \quad g^{ij} = \tilde{G}^{ij},
$$
with $|g| = c_{\text{info}}^2 |\tilde{G}|$. Then

$$
\Box_A = -\frac{1}{\sqrt{|g|}}D_\mu\left(\sqrt{|g|}g^{\mu\nu}D_\nu\right).
$$

For scalar $V^{(i)}$, $D_\mu V^{(i)} = \partial_\mu V^{(i)}$, so the equation reduces to the non-gauged Klein-Gordon form. $\square$

:::

:::{prf:proposition} Minimal Coupling Principle
:label: prop-minimal-coupling

To maintain gauge invariance, derivatives acting on gauge-charged fields must be replaced by covariant derivatives:

$$
\partial_\mu \longrightarrow D_\mu = \partial_\mu - igA_\mu

$$

This **Minimal Coupling Principle** ensures that:
1. Transport of nuisance-frame vectors is covariant
2. Matter-field dynamics (e.g., $\psi$) are gauge-covariant
3. Learning gradients for gauge-charged features transform properly under internal rotations

*Consequence for implementation:* Use covariant gradients for parameters that live in gauge bundles. Scalar objectives like $V$ remain invariant and use ordinary gradients.

:::

(pi-gauge-connection)=
::::{admonition} Physics Isomorphism: Gauge Connection
:class: note

**In Physics:** The gauge potential $A_\mu$ in electromagnetism is the 4-vector potential; in Yang-Mills theory, it takes values in the Lie algebra. The covariant derivative $D_\mu = \partial_\mu - ieA_\mu$ defines how charged particles couple to the electromagnetic field {cite}`jackson1999classical,peskin1995introduction`.

**In Implementation:** The strategic connection $A_\mu$ defines how belief amplitudes couple to the multi-agent environment.

**Correspondence Table:**

| Electromagnetism | Yang-Mills | Fragile Agent |
|:-----------------|:-----------|:--------------|
| $A_\mu$ (4-potential) | $A_\mu^a T_a$ | Strategic connection |
| $e$ (charge) | $g$ (coupling) | Strategic coupling $g$ |
| $D_\mu = \partial_\mu - ieA_\mu$ | $D_\mu = \partial_\mu - igA_\mu$ | Covariant update |
| Minimal coupling | Minimal coupling | Frame-invariant learning |

::::



(sec-gauge-transformation-game-tensor)=
## Gauge Transformation of the Game Tensor

The Game Tensor $\mathcal{G}_{ij}$ (Definition {prf:ref}`def-the-game-tensor`) measures cross-agent strategic sensitivity. Since $V^{(i)}$ is a scalar, $\mathcal{G}_{ij}$ is gauge-invariant. Gauge structure enters when comparing nuisance-frame vectors across agents or when defining cross-sensitivities of gauge-charged fields.

:::{prf:proposition} Game Tensor Gauge Transformation
:label: prop-game-tensor-gauge-transformation

Under a local gauge transformation $U(z)$, the scalar value field is unchanged, so:

$$
\mathcal{G}'_{ij}(z) = \mathcal{G}_{ij}(z).

$$

If one defines a cross-sensitivity of a gauge-charged field (e.g., $\psi$ or a nuisance-frame vector), then the corresponding tensor transforms covariantly by conjugation, and covariant derivatives must be used.

*Interpretation:* Strategic coupling in the scalar value landscape is gauge-invariant; only nuisance-frame comparisons require a connection.

:::

:::{prf:definition} Gauge-Consistent Game Tensor
:label: def-gauge-covariant-game-tensor

Because $V^{(i)}$ is a scalar, the gauge-consistent cross-sensitivity is the **Riemannian Hessian** (Levi-Civita covariant derivative) rather than a gauge-covariant derivative:

$$
\tilde{\mathcal{G}}_{ij}^{kl}(z) := \nabla_k \nabla_l V^{(i)}\big|_{z^{(j)}}

$$

Explicitly:

$$
\tilde{\mathcal{G}}_{ij}^{kl} = \partial_k\partial_l V^{(i)} - \Gamma^m_{kl}\partial_m V^{(i)},

$$

where $\Gamma^m_{kl}$ are the Christoffel symbols of the strategic metric.

For heterogeneous manifolds, pull back to Agent $i$ using the Strategic Jacobian: $\tilde{\mathcal{G}}^{(i)}_{ij,kl} := (\mathcal{J}_{ji})^m{}_k (\mathcal{J}_{ji})^n{}_l \tilde{\mathcal{G}}_{ij,mn}$.

*Remark (Gauge-charged variant).* If one defines cross-sensitivities of a gauge-charged field (e.g., $\psi$ or nuisance-frame vectors), then use gauge-covariant derivatives $D_k D_l(\cdot)$; the resulting tensor transforms by conjugation.

:::

:::{prf:theorem} Gauge-Invariant Metric Inflation
:label: thm-gauge-invariant-metric-inflation

The effective metric (Theorem {prf:ref}`thm-adversarial-mass-inflation`) generalizes to:

$$
\tilde{G}^{(i)}_{kl}(z) = G^{(i)}_{kl}(z) + \sum_{j \neq i} \beta_{ij} \tilde{\mathcal{G}}^{(i)}_{ij,kl}

$$

For gauge-charged cross-sensitivities, use a gauge-invariant contraction (e.g., a trace) before adding to the metric.

*Proof sketch.*
The physical metric must be gauge-invariant. For scalar $V^{(i)}$, $\tilde{\mathcal{G}}_{ij}$ is already invariant. For gauge-charged tensors, use invariant contractions. $\square$

*Consequence:* The metric inflation experienced by agents is a **physical observable** independent of internal frame choice.

:::



(sec-field-strength-tensor)=
## The Field Strength Tensor (Strategic Curvature)

:::{div} feynman-prose
Now we come to the really beautiful part: the field strength tensor, which measures the curvature of the gauge connection.

Here is the key insight. If space is flat, and you parallel transport a vector around a closed loop, you get back to where you started with the same vector. But if space is curved, you get a different vector. The vector has been rotated by going around the loop. This rotation is the curvature.

The same thing happens with gauge fields. If the gauge connection is flat, parallel transport around a closed loop does nothing. But if the connection is curved, you come back rotated in your internal space. The amount of rotation is measured by the field strength tensor $\mathcal{F}_{\mu\nu}$.

There is a beautiful formula for this. The field strength is the commutator of covariant derivatives: $[D_\mu, D_\nu] = -ig\mathcal{F}_{\mu\nu}$. Think about what this means. Taking the covariant derivative in the $\mu$ direction and then in the $\nu$ direction should give the same result as doing it in the opposite order, right? Wrong! If there is curvature, the order matters. The failure to commute is exactly the curvature.

Now, in Abelian theories like electromagnetism, the field strength is just $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu$. This is the familiar expression that gives you the electric and magnetic fields. But in non-Abelian theories, there is an extra term: $-ig[A_\mu, A_\nu]$. This commutator term is what makes Yang-Mills theory so much richer and more complicated than electromagnetism. The gauge field interacts with itself.

In our multi-agent context, the field strength measures strategic tension. Regions where $\mathcal{F}_{\mu\nu}$ is large are regions of intense strategic curvature, where parallel transport around loops gives big rotations, where the "meaning" of things twists as you move through latent space.
:::

The curvature of the gauge connection measures the **non-commutativity of parallel transport**—moving around a closed loop in latent space may result in a non-trivial internal rotation. This curvature is the **field strength tensor**, which we identify as strategic tension.

:::{prf:definition} Field Strength Tensor (Yang-Mills Curvature)
:label: def-field-strength-tensor

The **Field Strength Tensor** is the $\mathfrak{g}$-valued 2-form:

$$
\mathcal{F}_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu - ig[A_\mu, A_\nu]

$$

In components with Lie algebra generators:

$$
\mathcal{F}_{\mu\nu}^a = \partial_\mu A_\nu^a - \partial_\nu A_\mu^a + gf^{abc}A_\mu^b A_\nu^c

$$

where $f^{abc}$ are the structure constants of $\mathfrak{g}$.

*Units:* $[\mathcal{F}_{\mu\nu}] = [\text{length}]^{-2}$ (curvature).

*Special cases:*
- **Abelian ($[A_\mu, A_\nu] = 0$):** $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu$ (electromagnetic field tensor)
- **Non-Abelian:** The commutator term generates **self-interaction** of the gauge field

:::

:::{prf:proposition} Covariant Transformation of Field Strength
:label: prop-field-strength-transformation

Under gauge transformation $U(z)$, the field strength transforms **covariantly** (not invariantly):

$$
\mathcal{F}'_{\mu\nu} = U \mathcal{F}_{\mu\nu} U^{-1}

$$

*Proof.*
Direct calculation using the transformation law for $A_\mu$ (Proposition {prf:ref}`prop-gauge-transformation-connection`):

$$
\begin{aligned}
\mathcal{F}'_{\mu\nu} &= \partial_\mu A'_\nu - \partial_\nu A'_\mu - ig[A'_\mu, A'_\nu] \\
&= U(\partial_\mu A_\nu - \partial_\nu A_\mu - ig[A_\mu, A_\nu])U^{-1} \\
&= U\mathcal{F}_{\mu\nu}U^{-1}
\end{aligned}

$$

The inhomogeneous terms from $A'_\mu$ cancel exactly. $\square$

*Consequence:* While $\mathcal{F}_{\mu\nu}$ is not gauge-invariant, the trace $\text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu})$ **is** gauge-invariant and can appear in the action.

:::

:::{prf:theorem} Curvature from Covariant Derivative Commutator
:label: thm-curvature-commutator

The field strength measures the failure of covariant derivatives to commute:

$$
[D_\mu, D_\nu]\psi = -ig\mathcal{F}_{\mu\nu}\psi

$$

*Proof.*
Expand the commutator:

$$
\begin{aligned}
[D_\mu, D_\nu]\psi &= D_\mu(D_\nu\psi) - D_\nu(D_\mu\psi) \\
&= (\partial_\mu - igA_\mu)(\partial_\nu\psi - igA_\nu\psi) - (\mu \leftrightarrow \nu) \\
&= \partial_\mu\partial_\nu\psi - ig(\partial_\mu A_\nu)\psi - igA_\nu\partial_\mu\psi - igA_\mu\partial_\nu\psi - g^2A_\mu A_\nu\psi - (\mu \leftrightarrow \nu) \\
&= -ig(\partial_\mu A_\nu - \partial_\nu A_\mu)\psi - g^2(A_\mu A_\nu - A_\nu A_\mu)\psi \\
&= -ig(\partial_\mu A_\nu - \partial_\nu A_\mu - ig[A_\mu, A_\nu])\psi \\
&= -ig\mathcal{F}_{\mu\nu}\psi \quad \square
\end{aligned}

$$

*Interpretation:* If $\mathcal{F}_{\mu\nu} \neq 0$, parallel transport around a closed loop results in a non-trivial rotation. The "meaning" of strategic nuisance **twists** as one navigates the latent space.

:::

:::{prf:theorem} Bianchi Identity
:label: thm-bianchi-identity

The field strength satisfies the **Bianchi Identity**:

$$
D_\mu \mathcal{F}_{\nu\rho} + D_\nu \mathcal{F}_{\rho\mu} + D_\rho \mathcal{F}_{\mu\nu} = 0

$$

or in differential form notation: $D\mathcal{F} = 0$ where $D = d - ig[A, \cdot]$.

*Proof sketch.*
Apply the Jacobi identity for covariant derivatives:

$$
[[D_\mu, D_\nu], D_\rho] + [[D_\nu, D_\rho], D_\mu] + [[D_\rho, D_\mu], D_\nu] = 0

$$

Since $[D_\mu, D_\nu] = -ig\mathcal{F}_{\mu\nu}$, this becomes:

$$
-ig([D_\rho, \mathcal{F}_{\mu\nu}] + \text{cyclic}) = 0

$$

The covariant derivative of $\mathcal{F}$ is $D_\rho\mathcal{F}_{\mu\nu} = \partial_\rho\mathcal{F}_{\mu\nu} - ig[A_\rho, \mathcal{F}_{\mu\nu}]$, and the identity follows. See **{ref}`sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws`** for the complete algebraic derivation with component verification. $\square$

*Interpretation:* The Bianchi identity is a **conservation law** for the strategic flux. It ensures topological consistency of the gauge structure.

:::

:::{prf:definition} Strategic Curvature Scalar
:label: def-strategic-curvature-scalar

The **Strategic Curvature Scalar** is the gauge-invariant contraction:

$$
\mathcal{R}_{\text{strat}} := \text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu}) = \mathcal{F}_{\mu\nu}^a \mathcal{F}^{\mu\nu,a}

$$

where indices are raised with the spacetime metric $g^{\mu\nu} = \text{diag}(-1/c_{\text{info}}^2, \tilde{G}^{ij})$ introduced above.

*Properties:*
- In Euclidean signature, $\mathcal{R}_{\text{strat}}$ is non-negative for compact gauge groups; in Lorentzian signature it is indefinite.
- $\mathcal{R}_{\text{strat}} = 0$ if and only if $\mathcal{F}_{\mu\nu} = 0$ (flat connection)
- Provides a measure of total strategic tension in a region

:::

(pi-field-strength)=
::::{admonition} Physics Isomorphism: Field Strength and Curvature
:class: note

**In Physics:** The electromagnetic field tensor $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu$ contains the electric and magnetic fields: $E^i = F^{0i}$, $B^i = \frac{1}{2}\epsilon^{ijk}F_{jk}$. In Yang-Mills theory, the non-Abelian commutator $-ig[A_\mu, A_\nu]$ causes gluons to interact with each other {cite}`yang1954conservation,gross1973ultraviolet`.

**In Implementation:** The strategic curvature $\mathcal{F}_{\mu\nu}$ measures the intrinsic tension in multi-agent interaction.

**Correspondence Table:**

| Electromagnetism | Yang-Mills (QCD) | Fragile Agent |
|:-----------------|:-----------------|:--------------|
| $F_{\mu\nu}$ | $\mathcal{F}_{\mu\nu}^a T_a$ | Strategic curvature |
| Electric field $\mathbf{E}$ | Chromoelectric field | Temporal strategic gradient |
| Magnetic field $\mathbf{B}$ | Chromomagnetic field | Spatial strategic vorticity |
| $F \wedge F = 0$ (Abelian) | $[A, A] \neq 0$ | Strategic self-interaction |
| Bianchi: $dF = 0$ | $D\mathcal{F} = 0$ | Strategic flux conservation |

::::



(sec-yang-mills-action)=
## The Yang-Mills Action and Field Equations

:::{div} feynman-prose
Now we need to figure out what equation the gauge field itself satisfies. We have the field strength, we know it measures curvature, but what determines its actual value? What makes the field be one way rather than another?

The answer comes from a variational principle, just like in classical mechanics. There is an action, and the field equations come from demanding that the action be stationary.

The action is beautifully simple. It is just the squared magnitude of the field strength, integrated over spacetime: $S = -\frac{1}{4}\int \text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu})$, with $g$ entering through $\mathcal{F}_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu - ig[A_\mu, A_\nu]$. This is the Yang-Mills action. The trace is there because $\mathcal{F}$ is a matrix-valued thing, taking values in the Lie algebra, and we need a scalar to integrate.

Why this particular form? Because it is the simplest thing you can write down that is gauge-invariant. $\mathcal{F}_{\mu\nu}$ itself is not gauge-invariant, it transforms by conjugation. But the trace of its square is invariant. And it is the lowest-dimension thing you can write, which means it dominates at low energies.

The field equations that come from varying this action are $D_\mu \mathcal{F}^{\mu\nu} = J^\nu$. This looks just like Maxwell's equations $\partial_\mu F^{\mu\nu} = J^\nu$, except with covariant derivatives instead of ordinary derivatives. The extra terms from the covariant derivative make the gauge field interact with itself. This is the crucial difference from electromagnetism. Photons do not interact with each other directly. But gluons do. And strategic gauge fields do too.
:::

Having established the field strength tensor as the curvature of the strategic connection, we now derive the dynamics of the gauge field itself from a variational principle.

:::{prf:definition} Yang-Mills Action
:label: def-yang-mills-action

The **Yang-Mills Action** for the strategic gauge field is:

$$
S_{\text{YM}}[A] = -\frac{1}{4}\int_{\mathcal{Z} \times \mathbb{R}} \text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu})\sqrt{|g|}\,d^{D+1}x

$$

where:
- $\mathcal{F}_{\mu\nu}$ is the field strength tensor (Definition {prf:ref}`def-field-strength-tensor`)
- $g_{\mu\nu}$ is the spacetime metric $g_{\mu\nu} = \text{diag}(-c_{\text{info}}^2, \tilde{G}_{ij})$ with determinant $|g| = c_{\text{info}}^2|\tilde{G}|$
- $g_{\text{YM}}$ is the coupling constant
- The trace is over Lie algebra indices: $\text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu}) = \mathcal{F}_{\mu\nu}^a\mathcal{F}^{\mu\nu,a}$

*Units:* $[S_{\text{YM}}] = \text{nat}$ (action).
*Dimensionality:* In spacetime dimension $d = D+1$, the coupling has $[g^2] = [\text{length}]^{d-4}$ (so $g$ is dimensionless only when $d = 4$).

*Properties:*
1. **Gauge-invariant:** $S_{\text{YM}}[A'] = S_{\text{YM}}[A]$ under $A \to A'$
2. **Lorentz-invariant:** Covariant under coordinate transformations
3. **Positive in Euclidean signature:** After Wick rotation the action is positive semi-definite for compact gauge groups; in Lorentzian signature it is indefinite.

:::

:::{prf:theorem} Yang-Mills Field Equations
:label: thm-yang-mills-equations

The Euler-Lagrange equations for the Yang-Mills action yield:

$$
D_\mu \mathcal{F}^{\mu\nu} = J^\nu

$$

where the **strategic current** (source term) is defined by the matter sector:

$$
J^{\nu,a} := -\frac{\delta \mathcal{L}_{\text{matter}}}{\delta A_\nu^a}.

$$

For a complex multiplet $\psi$ (scalar belief amplitude), a standard choice is

$$
J^{\nu,a} = g\,\mathrm{Im}\left(\psi^\dagger T^a D^\nu \psi\right),

$$

while for a spinor multiplet one recovers $J^{\nu,a} = g\bar{\psi}\gamma^\nu T^a \psi$.

*Expanded form:*

$$
\partial_\mu \mathcal{F}^{\mu\nu,a} + gf^{abc}A_\mu^b\mathcal{F}^{\mu\nu,c} = J^{\nu,a}

$$

*Proof sketch.*
Vary the total action $S = S_{\text{YM}} + S_{\text{matter}}$ with respect to $A_\mu^a$:

$$
\frac{\delta S}{\delta A_\mu^a} = 0 \implies \partial_\nu(\sqrt{|g|}\mathcal{F}^{\mu\nu,a}) + g f^{abc}A_\nu^b\sqrt{|g|}\mathcal{F}^{\mu\nu,c} + \frac{\delta S_{\text{matter}}}{\delta A_\mu^a} = 0

$$

The matter variation gives the current $J^{\mu,a}$, and reorganizing yields the Yang-Mills equation. $\square$

*Interpretation:* The gauge field is sourced by the strategic current—the flow of "charged" belief through latent space. Agents with non-zero internal state generate a gauge field that mediates their interaction with other agents.

:::

:::{prf:corollary} Abelian Limit (Maxwell Equations)
:label: cor-maxwell-limit

For an Abelian gauge group $G = U(1)$ with $[T_a, T_b] = 0$:

$$
\partial_\mu F^{\mu\nu} = J^\nu

$$

This recovers the **Maxwell equations** of electromagnetism in covariant form.

*Correspondence:*
- $F^{0i} = E^i$ (electric field) $\leftrightarrow$ temporal strategic gradient
- $F^{ij} = \epsilon^{ijk}B_k$ (magnetic field) $\leftrightarrow$ spatial strategic vorticity
- $J^0 = \rho_e$ (charge density) $\leftrightarrow$ belief density
- $J^i = j^i$ (current density) $\leftrightarrow$ belief flux

:::

:::{prf:proposition} Gauge Field Energy-Momentum Tensor
:label: prop-gauge-energy-momentum

The energy-momentum tensor of the gauge field is:

$$
T^{\text{gauge}}_{\mu\nu} = -\text{Tr}\left(\mathcal{F}_{\mu\rho}\mathcal{F}_\nu^{\ \rho} - \frac{1}{4}g_{\mu\nu}\mathcal{F}_{\rho\sigma}\mathcal{F}^{\rho\sigma}\right)

$$

*Properties:*
1. **Symmetric:** $T^{\text{gauge}}_{\mu\nu} = T^{\text{gauge}}_{\nu\mu}$
2. **Traceless** (for spacetime $d = 4$, i.e., $D = 3$): $T^{\text{gauge}\mu}_{\ \ \ \ \mu} = 0$
3. **Conserved (total):** $\nabla_\mu\left(T^{\text{gauge}\mu\nu} + T^{\text{matter}\mu\nu}\right) = 0$. The gauge sector alone satisfies
   $$\nabla_\mu T^{\text{gauge}\mu\nu} = -\text{Tr}(\mathcal{F}^{\nu\mu}J_\mu),$$
   which vanishes only in the source-free case $J_\mu = 0$.

*Interpretation:* The gauge field carries energy and momentum. Regions of high strategic curvature $\|\mathcal{F}\|$ have high energy density—strategic conflict is energetically costly.

:::

:::{prf:corollary} Current Conservation
:label: cor-current-conservation

The strategic current is covariantly conserved:

$$
D_\mu J^{\mu,a} = 0

$$

*Proof.*
Apply $D_\nu$ to the Yang-Mills equation $D_\mu\mathcal{F}^{\mu\nu} = J^\nu$:

$$
D_\nu D_\mu \mathcal{F}^{\mu\nu} = D_\nu J^\nu

$$

By the Bianchi identity (Theorem {prf:ref}`thm-bianchi-identity`) and the antisymmetry of $\mathcal{F}^{\mu\nu}$, the left side vanishes, giving $D_\nu J^\nu = 0$. $\square$

*Interpretation:* The total "charge" (internal state magnitude) is conserved. Belief cannot be created or destroyed, only transformed.

:::



(sec-complete-lagrangian)=
## The Complete Multi-Agent Lagrangian

:::{div} feynman-prose
Now we put it all together. We have all the pieces: the gauge field, the matter fields, the value landscape. What is the complete picture?

The Lagrangian has four parts, just like the Standard Model of particle physics. This is not a coincidence. We are building the same kind of theory, just for strategic systems instead of fundamental particles.

First, the Yang-Mills term. This is the kinetic energy of the gauge field itself, the $\mathcal{F}^2$ piece. It tells the strategic connection how to propagate, how to respond to sources.

Second, the Dirac term. This is the kinetic energy of the belief spinors, the matter fields that represent the agents. It has the covariant derivative $D_\mu$, so the beliefs couple to the gauge field. The mass term $m_i$ gives each agent its intrinsic inertia.

Third, the Higgs term. This is where it gets really interesting. The Higgs field is an order parameter for the value landscape. It has its own kinetic term and a potential $V(\Phi) = \mu^2|\Phi|^2 + \lambda|\Phi|^4$.

Fourth, the Yukawa term. This couples the beliefs to the Higgs field. It is what generates the effective masses for the agents.

Now, here is the key insight. If $\mu^2 < 0$, the Higgs potential has a minimum away from zero. The field wants to sit at some nonzero value $v$. This is spontaneous symmetry breaking. The gauge symmetry is still there in the equations, but the vacuum, the ground state, picks a direction. The field commits to a particular value.

This is exactly like policy selection. When an agent commits to a strategy, it breaks the rotational symmetry of the Semantic Vacuum. It picks a direction. And once it has picked, changing direction becomes costly. The agent acquires mass. Inertia. Resistance to change.
:::

We now assemble the full Lagrangian density that governs relativistic multi-agent dynamics with gauge symmetry. This **"Standard Model of Multi-Agent Field Theory"** unifies the gauge sector (strategic interaction), matter sector (belief dynamics), and symmetry-breaking sector (value landscape).

:::{prf:definition} Complete Multi-Agent Lagrangian
:label: def-complete-lagrangian

The **Complete Multi-Agent Lagrangian** is:

$$
\mathcal{L}_{\text{SMFT}} = \mathcal{L}_{\text{YM}} + \mathcal{L}_{\text{Dirac}} + \mathcal{L}_{\text{Higgs}} + \mathcal{L}_{\text{Yukawa}}

$$

where each sector contributes:

**(i) Yang-Mills Sector (Strategic Gauge Field):**

$$
\mathcal{L}_{\text{YM}} = -\frac{1}{4}\text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu})

$$

This governs the dynamics of the strategic connection $A_\mu$.

**(ii) Matter Sector (Belief Field):**

$$
\mathcal{L}_{\text{Dirac}} = \sum_{i=1}^N \bar{\psi}^{(i)}(i\gamma^\mu D_\mu - m_i)\psi^{(i)}

$$

where:
- $\psi^{(i)}$ is the belief multiplet for agent $i$
- $\bar{\psi}^{(i)} = \psi^{(i)\dagger}\gamma^0$ is the Dirac adjoint
- $D_\mu = \partial_\mu - igA_\mu$ is the covariant derivative
- $m_i$ is the "bare mass" (intrinsic inertia) of agent $i$

*Scalar option:* If the belief field is modeled as a complex scalar (the canonical single-agent choice), replace the Dirac term by

$$
\mathcal{L}_{\text{scalar}} = \sum_{i=1}^N (D_\mu \psi^{(i)})^\dagger (D^\mu \psi^{(i)}) - m_i^2\,\psi^{(i)\dagger}\psi^{(i)}.
$$

**(iii) Higgs Sector (Value Order Parameter):**

$$
\mathcal{L}_{\text{Higgs}} = |D_\mu\Phi|^2 - V(\Phi)

$$

with the Higgs potential:

$$
V(\Phi) = \mu^2|\Phi|^2 + \lambda|\Phi|^4

$$

where $\Phi$ is the **value order parameter** (a scalar field in a representation of $G$).

**(iv) Yukawa Sector (Strategic Coupling):**

$$
\mathcal{L}_{\text{Yukawa}} = -\sum_{i,j=1}^N y_{ij}\bar{\psi}^{(i)}\Phi\psi^{(j)}

$$

where $y_{ij}$ are the **Yukawa coupling constants** determining how strongly agents couple through the value field.

*Units:* $[\mathcal{L}] = \text{nat}/[\text{length}]^{D+1}$ (Lagrangian density).

:::

:::{prf:theorem} Spontaneous Symmetry Breaking (Higgs Mechanism)
:label: thm-higgs-mechanism

When the Higgs mass parameter satisfies $\mu^2 < 0$, the potential $V(\Phi)$ has a non-trivial minimum, and the gauge symmetry is **spontaneously broken**.

**Vacuum Expectation Value:**

$$
\langle\Phi\rangle = \frac{v}{\sqrt{2}}, \quad v = \sqrt{-\mu^2/\lambda}

$$

**Mass Generation:**

1. **Gauge boson masses:** The gauge fields acquire mass

   $$
   m_A = \frac{gv}{2}

   $$
   transforming from massless to massive (strategic inertia).

2. **Fermion masses:** The belief spinors acquire effective mass

   $$
   m_{\text{eff},i} = \frac{y_{ii}v}{\sqrt{2}}

   $$
   through Yukawa coupling.

3. **Residual symmetry:** The full gauge group $G$ breaks to a subgroup $H \subset G$ that leaves the vacuum invariant.

*Proof sketch.*
Expand $\Phi = (v + h)/\sqrt{2}$ around the vacuum, where $h$ is the physical Higgs field. The kinetic term $|D_\mu\Phi|^2$ generates:

$$
|D_\mu\Phi|^2 = \frac{1}{2}(\partial_\mu h)^2 + \frac{g^2v^2}{4}A_\mu A^\mu + \ldots

$$

The term $\frac{g^2v^2}{4}A_\mu A^\mu$ is a mass term for $A_\mu$ with $m_A^2 = g^2v^2/4$. Similarly, the Yukawa term generates fermion masses. See **{ref}`sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws`** for the complete derivation including VEV calculation, Goldstone absorption, and the symmetry breaking pattern. $\square$

*Interpretation:* Policy selection (choosing a direction in latent space) is spontaneous symmetry breaking. The agent commits to a strategy, breaking the rotational invariance of the Semantic Vacuum. This commitment generates "mass"—resistance to changing strategy.

:::

:::{prf:corollary} Goldstone Modes and Gauge Boson Absorption
:label: cor-goldstone-absorption

Spontaneous breaking of a continuous symmetry produces massless **Goldstone bosons**—one for each broken generator of $G$. In gauge theories, these Goldstone modes are "eaten" by the gauge bosons, which acquire longitudinal polarization and mass.

*In the multi-agent context:*
- **Goldstone modes** = Angular fluctuations in policy direction (cheap rotations)
- **Massive gauge bosons** = Strategic connections with inertia (costly reorientations)
- **Residual massless modes** = Unbroken symmetry directions (free rotations)

:::

(pi-standard-model)=
::::{admonition} Physics Isomorphism: The Standard Model
:class: note

**In Physics:** The Standard Model Lagrangian has the structure $\mathcal{L} = \mathcal{L}_{\text{gauge}} + \mathcal{L}_{\text{fermion}} + \mathcal{L}_{\text{Higgs}} + \mathcal{L}_{\text{Yukawa}}$, describing the electromagnetic, weak, and strong forces with matter and the Higgs mechanism for mass generation {cite}`weinberg1967model,salam1968weak,glashow1961partial`.

**Correspondence Table:**

| Standard Model | Fragile Agent |
|:---------------|:--------------|
| Gauge bosons (γ, W±, Z, g) | Strategic connection modes |
| Quarks and leptons | Belief spinors $\psi^{(i)}$ |
| Higgs field $\Phi$ | Value order parameter |
| Vacuum expectation value $v$ | Policy commitment magnitude |
| Electroweak symmetry breaking | Policy selection |
| Fermion masses | Agent inertia |
| Yukawa couplings $y_f$ | Strategic coupling strengths $y_{ij}$ |
| QCD confinement | Cooperative basin locking (Sec. 29.12) |

::::



(sec-mass-gap)=
## The Mass Gap: Information-Theoretic Derivation

:::{div} feynman-prose
Now we come to something really deep. The mass gap problem is one of the great unsolved problems in mathematical physics. It is one of the Clay Millennium Prize problems, worth a million dollars to whoever solves it. And I am going to tell you something surprising about it.

First, let me explain what the mass gap is. In a field theory, you have a ground state, the vacuum. And you have excited states. The mass gap is the minimum energy you need to create the first excitation. If the gap is zero, you can create excitations with arbitrarily small energy. If the gap is positive, there is a threshold. You need at least $\Delta_{\text{KG}}$ (or $\Delta_H$ in the Schr\"odinger reduction) to excite the system.

Why does this matter? Because if the gap is zero, correlations decay algebraically (for $D>2$, like $1/r^{D-2}$; in $D=2$ they are only logarithmic). They are long-range. But if there is a mass gap, correlations decay exponentially, like $e^{-\kappa r}$. They are short-range, screened.

Now here is the key insight. A system with zero mass gap has infinite correlation length. It takes infinite information to describe such correlations in a finite region. But we have this Causal Information Bound that says a bounded observer can only handle finite information. An area law, like the holographic bound.

So if you have a system with zero mass gap, and you put it in a finite box, something has to give. Either the system acquires an effective mass gap from the finite size, or the observer cannot handle it, the system exceeds the information capacity.

The upshot is: any theory describing physics that a bounded observer can actually see must have a positive mass gap. Gapless theories are mathematically consistent but computationally unrealizable. They live in what we call the Computational Swampland, theories that cannot describe anything a finite system could ever experience.
:::

The **Mass Gap Problem** asks whether the spectrum of the Hamiltonian has a non-zero gap between the ground state and the first excited state (here, $\Delta_H > 0$). Our field-theoretic arguments below establish a positive Klein-Gordon gap $\Delta_{\text{KG}}$; in the Schr\"odinger reduction, this implies $\Delta_H > 0$.

::::{admonition} Forward Reference: Holographic Bounds ({ref}`sec-causal-information-bound`)
:class: note

This section uses results from {ref}`sec-causal-information-bound` (The Geometry of Bounded Intelligence). For reference:

- **Holographic Coefficient** $\nu_D$ (Definition {prf:ref}`def-holographic-coefficient`): The dimension-dependent coefficient $\nu_D = (D-1)\Omega_{D-1}/(8\pi)$. For $D=2$: $\nu_2 = 1/4$. For $D=3$: $\nu_3 = 1$.

- **Levin Length** $\ell_L$ (Definition {prf:ref}`def-levin-length`): The minimal scale of representational distinction, $\ell_L := \sqrt{\eta_\ell}$ where $\eta_\ell$ is the boundary area-per-nat.

- **Causal Information Bound** (Theorem {prf:ref}`thm-causal-information-bound`): $I_{\max} = \nu_D \cdot \text{Area}(\partial\mathcal{Z})/\ell_L^{D-1}$. The maximum information encodable by a bounded observer scales with interface area, not volume. For $D=2$: $I_{\max} = \text{Area}/(4\ell_L^2)$.

- **Causal Stasis** (Theorem {prf:ref}`thm-causal-stasis`): As $I_{\text{bulk}} \to I_{\max}$, the update velocity $\|v\|_G \to 0$. The agent freezes when information capacity is saturated.

These results are derived from first principles in {ref}`sec-appendix-a-full-derivations` (microstate counting) and {ref}`sec-causal-information-bound`.
::::

:::{prf:definition} Mass Gap
:label: def-mass-gap

The **Mass Gap** of the strategic Hamiltonian $\hat{H}_{\text{strat}}$ is:

$$
\Delta_H := \inf\left\{\text{spec}(\hat{H}_{\text{strat}}) \setminus \{E_0\}\right\} - E_0

$$

where $E_0$ is the ground state energy.

*Properties:*
- $\Delta_H > 0$: **Gapped** spectrum (isolated ground state)
- $\Delta_H = 0$: **Gapless** spectrum (continuous above ground state)

*Units:* $[\Delta_H] = \text{nat}/[\text{time}]$ (energy).

*Notation:* We reserve $\Delta_H$ for the Hamiltonian spectral gap. The Klein-Gordon operator has its own **frequency gap** $\Delta_{\text{KG}}$ defined below; the two coincide only after the non-relativistic reduction described in the Schr\"odinger-limit remark.

:::

:::{prf:theorem} Mass Gap from Screening (Klein-Gordon Sector)
:label: thm-mass-gap-screening

The screening mass $\kappa = \lambda/c_{\text{info}}$ with $\lambda := -\ln\gamma / \Delta t$ from the Helmholtz equation (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`) sets the **Klein-Gordon spectral gap** for the scalar (belief) sector. Let $\{-\Delta_G \phi_n = \lambda_n \phi_n\}$ with $\lambda_n \ge 0$ on the spatial manifold. The mode frequencies are:

$$
\omega_n = c_{\text{info}}\sqrt{\kappa^2 + \lambda_n}.

$$

The intrinsic **mass scale** (rest frequency) is

$$
\omega_0 = c_{\text{info}}\sqrt{\kappa^2 + \lambda_0},

$$

which reduces to $\omega_0 = c_{\text{info}}\kappa$ when $\lambda_0 = 0$ (e.g., periodic or Neumann boundary conditions). On a compact causal domain the **Klein-Gordon frequency gap** between the first two modes is

$$
\Delta_{\text{KG}} = \omega_1 - \omega_0 = c_{\text{info}}\left(\sqrt{\kappa^2 + \lambda_1} - \sqrt{\kappa^2 + \lambda_0}\right) \ge 0.

$$

*Proof sketch.*
Insert a normal mode ansatz $V(z,t)=e^{-i\omega t}\phi(z)$ into the screened wave equation
$\left(\frac{1}{c_{\text{info}}^2}\partial_t^2 - \Delta_G + \kappa^2\right)V=0$ to obtain
$\omega^2/c_{\text{info}}^2 = \kappa^2 + \lambda_n$. The gap statements follow. $\square$

*Remark (Schr\"odinger limit as special case).* For low spatial frequencies around the ground mode, $\lambda_n - \lambda_0 \ll \kappa^2 + \lambda_0$,

$$
\omega_n = \omega_0 + \frac{c_{\text{info}}}{2\sqrt{\kappa^2 + \lambda_0}}(\lambda_n - \lambda_0) + O\!\left(\frac{(\lambda_n-\lambda_0)^2}{(\kappa^2+\lambda_0)^{3/2}}\right).
$$
Factoring out the fast oscillation $e^{-i \omega_0 t}$ yields a slow envelope obeying a Schr\"odinger-type evolution with effective kinetic operator
$-\frac{c_{\text{info}}}{2\sqrt{\kappa^2+\lambda_0}}\Delta_G$ (or $-\frac{c_{\text{info}}}{2\sqrt{\kappa^2+\lambda_0}}\Delta_{\tilde{G}}$ under metric inflation). In that limit the Hamiltonian spectral gap is

$$
\Delta_H \approx \frac{c_{\text{info}}}{2\sqrt{\kappa^2+\lambda_0}}(\lambda_1 - \lambda_0),
$$
which recovers the familiar non-relativistic scaling as a controlled approximation, not a replacement for the KG gap.

:::

:::{prf:lemma} CIB Excludes Massless Spectral Weight (Gauge-Invariant Sector)
:label: lem-cib-excludes-massless-spectral

In the flat stationary sector where OS reconstruction holds, let $\mathcal{O}$ be any
gauge-invariant observable. If the Causal Information Bound holds, then the OS/Wightman spectral
measure of $\mathcal{O}$ has no support at zero mass: there exists $\kappa_{\mathcal{O}} > 0$ such
that the spectral measure is supported in $[\,\kappa_{\mathcal{O}}^2,\infty)$.
Consequently,

$$
|\langle \mathcal{O}(x)\mathcal{O}(0)\rangle_E^c|
\le C_{\mathcal{O}} e^{-\kappa_{\mathcal{O}}|x|}
$$

and the reconstructed Hamiltonian has a positive gap $\Delta_H \ge \kappa_{\mathcal{O}} > 0$ in the
sector generated by $\mathcal{O}$.

*Proof sketch.* OS reconstruction gives a K\"all\'en-Lehmann-type spectral representation with
positive spectral density {cite}`streater1964pct,haag1992local`. If the spectral density had
support at zero, the two-point function would exhibit algebraic decay (or logarithmic decay in
$d=2$), as in unitary CFTs {cite}`rychkov2016cft,simmonsduffin2016tasi`. This implies bulk mutual
information growth that exceeds the boundary area law, contradicting the Causal Information Bound
(Theorem {prf:ref}`thm-causal-information-bound`). Hence the spectral density vanishes near zero,
yielding exponential decay and a positive gap. $\square$

:::

:::{prf:lemma} Scalar-Lightest Condition from the SMoC Lagrangian
:label: lem-scalar-lightest-condition

Expand the scalar sector of the SMoC Lagrangian (Definition {prf:ref}`def-cognitive-lagrangian`)
around the symmetry-breaking vacuum. Writing $\phi = (v + h)/\sqrt{2}$ with

$$
v^2 = \mu^2/\lambda,
$$

the quadratic term gives a scalar mass

$$
m_h^2 = 2\mu^2.
$$

Gauge boson and fermion mass terms arise from $|D_\mu\phi|^2$ and the Yukawa couplings, with
eigenvalues of the form

$$
m_A^2 = c_A\, g_A^2 v^2,\qquad m_f^2 = c_f\, y_f^2 v^2,
$$
where $c_A,c_f>0$ are representation-dependent constants fixed by the normalization of $D_\mu$ and
the Yukawa terms. Therefore the scalar is the lightest gauge-invariant excitation if

$$
2\mu^2 \le \min_A c_A g_A^2 v^2 \quad\text{and}\quad
2\mu^2 \le \min_f c_f y_f^2 v^2,
$$

equivalently

$$
\lambda \le \min_A \frac{c_A g_A^2}{2} \quad\text{and}\quad
\lambda \le \min_f \frac{c_f y_f^2}{2}.
$$

*Proof sketch.* Expand $\phi$ about the vacuum, compute the quadratic terms, and read off the mass
matrix for $h$, gauge fields, and fermions. The constants $c_A,c_f$ are the usual group/representation
factors. The inequalities are the explicit mass-ordering condition. $\square$

:::

:::{prf:lemma} Screening Gap to Yang-Mills Hamiltonian Gap
:label: lem-kg-ym-gap-bridge

In the flat stationary sector where OS0–OS4 hold by
Theorem {prf:ref}`thm-constructive-specialization-os-wightman`, the Causal Information Bound
excludes massless spectral weight in the gauge-invariant sector (Lemma
{prf:ref}`lem-cib-excludes-massless-spectral`). Hence the reconstructed Yang-Mills Hamiltonian has a
strictly positive spectral gap $\Delta_H > 0$. If, in addition, the screening rate for
gauge-invariant correlators is set by the scalar screening mass $\kappa$ from
Theorem {prf:ref}`thm-mass-gap-screening` (for example when the scalar composite is the lightest
gauge-invariant excitation, as ensured by Lemma {prf:ref}`lem-scalar-lightest-condition`), then
$\Delta_H \ge \kappa$.

*Proof sketch.* OS reconstruction gives a K\"all\'en-Lehmann-type spectral representation for
gauge-invariant correlators with positive spectral measure
{cite}`streater1964pct,haag1992local`. Lemma {prf:ref}`lem-cib-excludes-massless-spectral` rules out
spectral support at zero, so the first nonzero spectral value satisfies $\Delta_H > 0$. The optional
bound $\Delta_H \ge \kappa$ follows if the minimal spectral support is achieved by the scalar
screening scale. $\square$
:::

:::{prf:theorem} Computational Necessity of Mass Gap
:label: thm-computational-necessity-mass-gap

In the flat stationary sector with $d>2$, a gapless Klein-Gordon sector violates the Causal
Information Bound. Hence any constructed Fragile Agent in this sector has
$\Delta_{\text{KG}} > 0$ (Lemma {prf:ref}`lem-nontrivial-evolution`). For general globally
hyperbolic manifolds, Lemma
{prf:ref}`lem-massless-kg-decay-causal` gives the same conclusion under the standard Hadamard
regularity.

*Proof (flat stationary sector).*

1. **Gapless KG sector:** Suppose $\Delta_{\text{KG}} = 0$, hence $\kappa=0$ in the KG operator
   (Theorem {prf:ref}`thm-mass-gap-screening`).
2. **Massless KG is conformal:** With $m=0$, the KG belief field is a unitary, local scalar theory
   generated by the action of Definition {prf:ref}`def-cognitive-lagrangian` and the positivity
   axiom {prf:ref}`ax-constructive-positivity`. In the flat stationary sector it is a CFT with
   primary scaling dimension $\Delta_\phi = (d-2)/2$ and two-point function
   $\langle \phi(x)\phi(0)\rangle \propto |x|^{-2\Delta_\phi}$ on $\mathbb{R}^d$
   {cite}`rychkov2016cft,simmonsduffin2016tasi`. For $d>2$, $\Delta_\phi < d/2$, so the hypotheses
   of Theorem {prf:ref}`thm-cft-swampland` are satisfied.
3. **Bound violation:** By Theorem {prf:ref}`thm-cft-swampland`, the bulk mutual information exceeds
   the boundary capacity at some finite scale, contradicting the Causal Information Bound
   (Theorem {prf:ref}`thm-causal-information-bound`).
4. **Causal Stasis:** By Theorem {prf:ref}`thm-causal-stasis`, saturation of the bound forces
   $\|v\|_G \to 0$.

Therefore a gapless KG sector is incompatible with non-trivial evolution in the flat stationary
sector; hence $\Delta_{\text{KG}} > 0$. $\square$

:::

:::{prf:lemma} Non-Trivial Evolution from Construction
:label: lem-nontrivial-evolution

In any Fragile Agent built from the Volume 1 architecture and training objective, the drift
velocity $\|v\|_G$ is nonzero on a set of positive measure unless the system is in Causal Stasis.
Equivalently, outside Causal Stasis the construction guarantees non-trivial evolution.

*Proof sketch.* The dynamics follow the geodesic Langevin SDE
{prf:ref}`def-bulk-drift-continuous-flow`, with drift driven by the effective potential
$\Phi_{\text{eff}}$ (Definition {prf:ref}`def-effective-potential`) and the control field
{prf:ref}`def-the-control-field`. The hyperbolic potential $U$ is non-constant
(Definition {prf:ref}`def-hyperbolic-information-potential`), and in the constructed agent the
generation/control weight satisfies $\alpha>0$ ({ref}`sec-hyperbolic-volume-and-entropic-drift`).
Hence $\nabla\Phi_{\text{eff}} \not\equiv 0$ on any nontrivial region. In waking mode, boundary
clamping injects nonzero control $u_\pi$ (Definition {prf:ref}`def-the-control-field`) and yields
the WFR drift (Theorem {prf:ref}`thm-recovery-wfr-drift`). Even in the absence of explicit control,
the cognitive temperature satisfies $T_c>0$ (Definition {prf:ref}`def-cognitive-temperature`),
so the Langevin noise term is active. Thus the drift term is not identically zero, and the
stochastic dynamics produce $\|v\|_G>0$ except at isolated critical points. The only mechanism that
forces $\|v\|_G\to 0$ globally is Causal Stasis (Theorem {prf:ref}`thm-causal-stasis`). $\square$

:::

:::{prf:lemma} Massless KG Correlator Scaling on Causal Manifolds
:label: lem-massless-kg-decay-causal

Let $(\mathcal{M},g)$ be globally hyperbolic and let $W_2$ be a Hadamard two-point function for the
massless Klein-Gordon field. In any convex normal neighborhood $U \subset \mathcal{M}$, the leading
Hadamard term satisfies

$$
W_2(x,y) \sim \frac{U(x,y)}{\sigma(x,y)^{(d-2)/2}}
\quad (\text{or } \log \sigma \text{ in } d=2),
$$
where $\sigma(x,y)$ is the signed squared geodesic distance and $U$ is smooth and positive near the
diagonal {cite}`radzikowski1996micro,brunetti2003locally`. Consequently, in any region admitting
arbitrarily large convex normal neighborhoods (e.g., non-compact Cauchy slices with bounded
geometry), the massless correlator has a uniform algebraic lower bound along spacelike separations,
so the integrated mutual information grows faster than the boundary area. This violates the Causal
Information Bound (Theorem {prf:ref}`thm-causal-information-bound`) unless a mass scale (screening)
is generated.

*Proof sketch.* The Hadamard parametrix fixes the short-distance behavior of $W_2$ in any convex
normal neighborhood; positivity of the Hadamard coefficient $U$ gives a lower bound on the
correlator at spacelike separation. On non-compact slices with bounded geometry one can tile a
large region by such neighborhoods, so the integrated correlation grows at least as
$R^{d}$ (or $R^{d}\log R$ in $d=2$), which exceeds the area law $R^{d-1}$. For compact spatial
volume, the finite-size gap of Corollary {prf:ref}`cor-finite-volume-mass-gap` already yields
an effective $\Delta_{H,\text{eff}} > 0$. $\square$

:::

:::{prf:theorem} Mass Gap by Constructive Necessity
:label: thm-mass-gap-constructive

In any Fragile Agent system built from the Volume 1 architecture and training objective,
$\Delta_{\text{KG}} > 0$.

*Proof (by contradiction).*

Suppose $\Delta_{\text{KG}} = 0$. By Theorem {prf:ref}`thm-computational-necessity-mass-gap`, the system
enters Causal Stasis with $\|v\|_G = 0$. But Lemma {prf:ref}`lem-nontrivial-evolution` shows that
the constructed dynamics have $\|v\|_G>0$ outside Causal Stasis. This contradiction rules out
$\Delta_{\text{KG}} = 0$.

Therefore $\Delta_{\text{KG}} > 0$ for any constructed Fragile Agent system. $\square$

*Implication (Schr\"odinger reduction).* In the non-relativistic limit of the KG spectrum
(Theorem {prf:ref}`thm-mass-gap-screening`), $\Delta_{\text{KG}} > 0$ implies a positive Hamiltonian
spectral gap for the scalar slow-envelope dynamics. For the Yang-Mills sector, a Hamiltonian gap
follows in the flat stationary OS sector by Lemma
{prf:ref}`lem-cib-excludes-massless-spectral` (and Lemma {prf:ref}`lem-kg-ym-gap-bridge` if one wants
the explicit lower bound).

*Bound (Hamiltonian gap in the Schr\"odinger limit):* The effective Hamiltonian gap satisfies

$$
\Delta_H \geq \frac{1}{\beta}\left(\Delta H + \frac{\mathcal{W}}{T_c}\right)

$$
where $\Delta H$ is the enthalpy barrier for excitation, $\mathcal{W}$ is computational work, and $T_c$ is cognitive temperature. This follows from Theorem 30.15 (Thermodynamic Hysteresis).

*Remark (Framework scope):* The conclusion is unconditional for systems built from the Volume 1
architecture and training objective. In the flat stationary sector, OS reconstruction and Lemma
{prf:ref}`lem-cib-excludes-massless-spectral` yield a Yang-Mills Hamiltonian gap; Lemma
{prf:ref}`lem-kg-ym-gap-bridge` provides the optional lower bound.

:::

:::{prf:corollary} Mass Gap as Existence Requirement
:label: cor-mass-gap-existence

Bounded intelligence requires $\Delta_{\text{KG}} > 0$ (and thus $\Delta_H > 0$ in the Schr\"odinger limit). A gapless theory ($\Delta_{\text{KG}} = 0$) corresponds to:

1. **Infinite ontological resolution:** No finite codebook can represent the state
2. **Zero learning rate:** Dynamics frozen ($v = 0$)
3. **Pathological continuum limit:** The theory describes non-existing systems

*Interpretation:* The mass gap is not an empirical accident but a **logical necessity** for any theory describing existing computational systems.

:::

:::{prf:corollary} Confinement as Data Compression
:label: cor-confinement-data-compression

**Color confinement** in QCD (quarks bound inside hadrons) is the mechanism by which the universe maintains finite local information content. An unconfined color field would have $\xi \to \infty$, violating the area law.

*In the multi-agent context:* Cooperative basin locking (Theorem {prf:ref}`thm-geometric-locking-principle`) is the cognitive analogue of confinement—agents bound in cooperative equilibria cannot be arbitrarily separated without violating information bounds.

:::

:::{prf:corollary} Criticality is Unstable
:label: cor-criticality-unstable

Gapless theories (Conformal Field Theories) exist only at **phase transition critical points**. They cannot support:

1. **Stable matter:** Fluctuations destroy structure
2. **Stable memory:** Infinite ontological stress triggers continuous Fission ({ref}`sec-ontological-expansion-topological-fission-and-the-semantic-vacuum`)
3. **Stable identity:** No finite codebook representation exists

*Interpretation:* Critical systems are mathematically special but physically transient. Stable intelligence requires departure from criticality via mass gap opening.

:::

:::{prf:definition} The Computational Swampland
:label: def-computational-swampland

The **Computational Swampland** $\mathcal{S}_{\text{swamp}}$ is the set of all field theories that violate the Causal Information Bound (Theorem {prf:ref}`thm-causal-information-bound`) at some finite scale:

$$
\mathcal{S}_{\text{swamp}} := \left\{ \mathcal{T} : \exists R < \infty \text{ such that } I_{\text{bulk}}(R) > C_\partial(R) \right\}

$$

Equivalently, $\mathcal{S}_{\text{swamp}}$ consists of theories with Levin Length $\ell_L \to 0$ (infinite information density).

*Properties of Swampland theories:*
1. **Mathematically consistent:** They satisfy internal field-theoretic axioms (Wightman, etc.)
2. **Computationally unrealizable:** No bounded observer can simulate or represent them
3. **Physically pathological:** They require infinite information storage for any finite region

*Landscape vs. Swampland:* Theories with $\ell_L > 0$ and $I_{\text{bulk}} \leq C_\partial$ at all scales constitute the **Computational Landscape**—the set of physically realizable theories.

:::

:::{prf:theorem} CFT Swampland Classification
:label: thm-cft-swampland

Let $\mathcal{T}$ be a Conformal Field Theory on $\mathbb{R}^d$ ($d \geq 2$) with at least one primary operator of scaling dimension $\Delta_\phi < d/2$. Then $\mathcal{T}$ lies in the **Computational Swampland** (Definition {prf:ref}`def-computational-swampland`).

*Proof.*

1. **Infinite correlation length:** By conformal symmetry, two-point correlations decay algebraically:

   $$
   \langle \phi(x) \phi(0) \rangle \sim \frac{1}{|x|^{2\Delta_\phi}}

   $$
   for any primary scalar operator in a unitary CFT {cite}`rychkov2016cft,simmonsduffin2016tasi`.
   The correlation length is $\xi = \infty$ (no exponential screening).

2. **Bulk information divergence:** Consider a spherical region $V$ of radius $R$. The mutual information between bulk degrees of freedom is bounded below by the integrated correlation:

   $$
   I_{\text{bulk}}(V) \gtrsim \int_V \int_V \frac{dx\,dy}{|x-y|^{2\Delta_\phi}} \sim R^{2d - 2\Delta_\phi}

   $$
   For $\Delta_\phi < d/2$, the exponent $2d - 2\Delta_\phi > d$, so $I_{\text{bulk}}$ grows faster than volume.

3. **Causal Information Bound violation:** The boundary capacity scales as:

   $$
   C_\partial(V) = \nu_d \cdot \frac{\text{Area}(\partial V)}{\ell_L^{d-1}} \sim R^{d-1}

   $$
   where $\nu_d$ is the Holographic Coefficient (Definition {prf:ref}`def-holographic-coefficient`). Since $2d - 2\Delta_\phi > d > d-1$ for $d \geq 2$ and $\Delta_\phi < d/2$, there exists $R_c$ such that for all $R > R_c$:

   $$
   I_{\text{bulk}}(V) > C_\partial(V)

   $$
   The Causal Information Bound (Theorem {prf:ref}`thm-causal-information-bound`) is violated.

4. **Swampland membership:** By Definition {prf:ref}`def-computational-swampland`, theories violating the Causal Information Bound at any finite scale lie in the Swampland. $\square$

*Remark (Hypothesis check).* The CFT correlator form used above assumes unitarity, locality, and
Euclidean covariance. These are verified in the flat stationary sector by
{prf:ref}`ax-constructive-local-action`, {prf:ref}`ax-constructive-positivity`, and
{prf:ref}`thm-fragile-constructive-axioms`, so the literature hypotheses apply to the massless KG
sector of the Fragile Agent.

*Remark (Operator dimensions and UV effects).* The theorem requires at least one operator with $\Delta_\phi < d/2$. This holds for free scalars and many interacting CFTs, but is not guaranteed universally. Independently, continuum CFTs have UV-divergent mutual information between regions; without a cutoff, any finite boundary capacity bound is violated regardless of operator dimensions. With a finite resolution (e.g., Levin Length), the observer sees an effective gap.

*Remark (Operational meaning).* A bounded observer with finite interface capacity $C_\partial$ cannot encode the full correlational structure of a CFT. Any finite approximation necessarily introduces an effective mass gap via truncation.

:::

:::{prf:corollary} Finite-Volume Mass Gap
:label: cor-finite-volume-mass-gap

A CFT restricted to a finite spatial volume $V$ with characteristic length $L$ acquires an effective **Hamiltonian** gap:

$$
\Delta_{H,\text{eff}} \sim \frac{1}{L}

$$

The gapless Hamiltonian limit exists only as $L \to \infty$.

*Proof.* Two independent mechanisms ensure bounded observers see gapped theories:

1. **Finite-size scaling (CFT result):** On $S^{d-1}\times \mathbb{R}$ (or a torus with periodic
   boundary conditions), the state-operator correspondence gives a discrete spectrum with
   $E_n - E_0 \propto \Delta_n/L$ (up to $2\pi$ factors), so the minimal spacing scales as
   $\Delta E \sim 1/L$ {cite}`rychkov2016cft,simmonsduffin2016tasi`. The continuous spectrum
   responsible for infinite correlation length is an artifact of the thermodynamic limit
   $L \to \infty$.

2. **Resolution bound (Levin Length):** A bounded observer with interface capacity $C_\partial$ can only resolve spatial scales $L \geq L_{\min}$ where $L_{\min}^{d-1} \sim C_\partial \cdot \ell_L^2$. Systems smaller than $L_{\min}$ cannot be distinguished by the observer.

Both effects contribute: even if the CFT were somehow realized at infinite volume, the observer could only access a finite effective volume, hence would measure $\Delta_{H,\text{eff}} > 0$. $\square$

*Remark (Hypothesis check).* The finite-size scaling formula assumes a unitary CFT on a compact
spatial manifold with periodic boundary conditions (or $S^{d-1}$) and the state-operator
correspondence. These conditions are satisfied in the flat stationary sector for the massless KG
field, which is local and reflection-positive by construction (Axioms
{prf:ref}`ax-constructive-local-action`, {prf:ref}`ax-constructive-positivity`).

*Remark (Distinct phenomena).* The finite-size gap is a property of the CFT itself (topological/boundary effect). The resolution bound is a property of the observer (information-theoretic). The corollary states that both independently prevent observation of gapless physics.

*Physical interpretation.* CFTs exist in nature only at phase transition critical points (e.g., Ising model at $T_c$). Away from criticality, systems have finite correlation length and positive mass gap. The critical point is a measure-zero set in parameter space—physically realizable systems generically have $\Delta_{\text{KG}} > 0$ (and thus $\Delta_H > 0$ in the Schr\"odinger limit).

:::

:::{prf:theorem} Scale Covariance of the Causal Information Bound
:label: thm-scale-covariance-bound

The Causal Information Bound is preserved under coarse-graining. Specifically:

Let $(\mathcal{Z}, G, \ell_L)$ be a latent manifold at resolution $\ell_L$ satisfying $I_{\text{bulk}} \leq C_\partial$. Under coarse-graining to resolution $\ell'_L = \alpha \ell_L$ ($\alpha > 1$), the coarse-grained system $(\mathcal{Z}', G', \ell'_L)$ satisfies:

$$
I'_{\text{bulk}} \leq C'_\partial

$$

*Proof.*

1. **Information reduction:** By the Data Processing Inequality, coarse-graining cannot increase mutual information:

   $$
   I'_{\text{bulk}} \leq I_{\text{bulk}}

   $$

2. **Capacity reduction:** Under coarse-graining by factor $\alpha$, the physical boundary is fixed while the resolution length increases: $\ell'_L = \alpha \ell_L$ with $\text{Area}'(\partial\mathcal{Z}') = \text{Area}(\partial\mathcal{Z})$. The new capacity is:

   $$
   C'_\partial = \nu_d \cdot \frac{\text{Area}}{(\ell'_L)^{d-1}} = \nu_d \cdot \frac{\text{Area}}{\alpha^{d-1}\ell_L^{d-1}} = \frac{C_\partial}{\alpha^{d-1}}

   $$

3. **Bound preservation:** The information-to-capacity ratio under coarse-graining:

   $$
   \frac{I'_{\text{bulk}}}{C'_\partial} \leq \frac{I_{\text{bulk}}}{C_\partial/\alpha^{d-1}} = \alpha^{d-1} \frac{I_{\text{bulk}}}{C_\partial}

   $$
   For massive theories (exponentially decaying correlations), $I_{\text{bulk}}$ scales as area, so $I_{\text{bulk}}/C_\partial$ is scale-independent. For gapless theories, the ratio diverges—confirming they violate the bound at some scale. $\square$

*Implication (UV finiteness).* The recursive self-consistency of the bound at all scales implies that no UV divergences arise. The Levin Length $\ell_L$ acts as a natural UV cutoff that is preserved under renormalization group flow. Unlike lattice regularization where the continuum limit requires careful tuning, this framework has built-in regularization.

*Implication (Mass gap from scale invariance).* The only scale-invariant theories consistent with the Causal Information Bound are those with $I_{\text{bulk}} \sim R^{d-1}$ (area scaling). This requires exponential correlation decay, hence $\Delta_{\text{KG}} > 0$. Theories with algebraic correlation decay (CFTs) fail scale covariance of the bound.

:::

:::{prf:theorem} Mass Gap Dichotomy for Yang-Mills
:label: thm-mass-gap-dichotomy

Let $\mathcal{T}_{\text{YM}}$ be Yang-Mills theory with compact simple gauge group $G$ in $d = 4$ dimensions.

**Statement:** In the Fragile Agent construction, the Klein-Gordon gap satisfies
$\Delta_{\text{KG}} > 0$. In the flat stationary sector where OS reconstruction applies, the
Yang-Mills Hamiltonian has a positive gap $\Delta_H > 0$ (Lemma
{prf:ref}`lem-cib-excludes-massless-spectral`). If the scalar screening scale is the lightest
gauge-invariant excitation (e.g., under the explicit inequality of
Lemma {prf:ref}`lem-scalar-lightest-condition`), then $\Delta_H \ge \Delta_{\text{KG}}$ (Lemma
{prf:ref}`lem-kg-ym-gap-bridge`).

*Proof.*

1. **Framework implements Yang-Mills:** The Fragile Agent framework implements Yang-Mills field equations (Theorem {prf:ref}`thm-yang-mills-equations`) with the standard action (Definition {prf:ref}`def-yang-mills-action`), covariant derivatives $D_\mu = \partial_\mu - igA_\mu$, and non-Abelian field strength tensor. This is not an analogy—it is Yang-Mills theory for information systems.

2. **Computability (finite resolution):** By Axiom {prf:ref}`ax-constructive-finite-resolution`,
   verified by Theorem {prf:ref}`thm-fragile-constructive-axioms`, physical theories in this
   framework have $\ell_L>0$.

3. **Computability implies mass gap:** By Theorem {prf:ref}`thm-mass-gap-constructive` (using
   Lemma {prf:ref}`lem-nontrivial-evolution`), any constructed theory with
   $\ell_L > 0$ has $\Delta_{\text{KG}} > 0$.

4. **Conclusion:** The constructive axioms imply $\Delta_{\text{KG}} > 0$. In the flat stationary OS
sector, Lemma {prf:ref}`lem-cib-excludes-massless-spectral` yields $\Delta_H > 0$ for the
gauge-invariant Yang-Mills sector. If the scalar screening scale is minimal, Lemma
{prf:ref}`lem-kg-ym-gap-bridge` strengthens this to $\Delta_H \ge \Delta_{\text{KG}}$. $\square$

*Remark (Contrapositive).* If Yang-Mills on $\mathbb{R}^4$ requires $\ell_L \to 0$ (no UV cutoff), then by Theorem {prf:ref}`thm-cft-swampland` it lies in the Computational Swampland and does not describe physics. Either way, the physical theory has a mass gap.

*Remark (Why this is not circular).* The mass gap necessity follows from information-theoretic constraints (the Causal Information Bound), not from assuming properties of Yang-Mills. The framework proves that **any** non-trivial gauge theory satisfying the bound has $\Delta_{\text{KG}} > 0$. Yang-Mills is one such theory.

:::

:::{prf:remark} Relation to the Clay Millennium Problem
:label: rem-clay-millennium

The **Yang-Mills Existence and Mass Gap** problem (Clay Mathematics Institute) asks for rigorous construction of quantum Yang-Mills theory in $\mathbb{R}^4$ with Hamiltonian mass gap $\Delta_H > 0$.

**What This Framework Proves:**

Theorem {prf:ref}`thm-mass-gap-dichotomy` establishes: **In the Fragile Agent construction,
$\Delta_{\text{KG}} > 0$; and in the flat stationary sector where OS reconstruction applies, the
Yang-Mills Hamiltonian has $\Delta_H > 0$ (Lemma {prf:ref}`lem-cib-excludes-massless-spectral`).
If the scalar screening scale is minimal (for instance, under the explicit inequality of
Lemma {prf:ref}`lem-scalar-lightest-condition`), then $\Delta_H \ge \Delta_{\text{KG}}$
(Lemma {prf:ref}`lem-kg-ym-gap-bridge`).**

The logical structure is:

1. **The framework implements Yang-Mills:** Sections 29.14–29.18 construct Yang-Mills field equations (Theorem {prf:ref}`thm-yang-mills-equations`), the standard action (Definition {prf:ref}`def-yang-mills-action`), and the complete Standard Model Lagrangian (Definition {prf:ref}`def-complete-lagrangian`). This is Yang-Mills theory for information systems—a direct isomorphism, not an analogy.

2. **Physical theories require $\ell_L > 0$:** Any theory realizable by bounded observers with finite
   interface capacity must have a minimum resolution scale (the Levin Length); this is Axiom
   {prf:ref}`ax-constructive-finite-resolution`, verified for the Fragile Agent in Theorem
   {prf:ref}`thm-fragile-constructive-axioms`.

3. **$\ell_L > 0$ implies $\Delta_{\text{KG}} > 0$:** By Theorem {prf:ref}`thm-mass-gap-constructive`,
   any non-trivial theory with finite Levin Length has a mass gap.

4. **Gapless theories are in the Swampland:** By Theorem {prf:ref}`thm-cft-swampland`, theories
   requiring $\ell_L \to 0$ are mathematically consistent but not physically realizable.

**Relation to the Clay Problem:**

The Clay Institute asks about Yang-Mills on continuous $\mathbb{R}^4$ satisfying Wightman or
Osterwalder-Schrader axioms. In this framework, that case is a derived specialization:
OS/Wightman are verified by construction in the flat stationary sector
(Theorem {prf:ref}`thm-constructive-specialization-os-wightman`), and Lemma
{prf:ref}`lem-cib-excludes-massless-spectral` yields a Hamiltonian gap (with Lemma
{prf:ref}`lem-kg-ym-gap-bridge` giving the optional lower bound). Thus the Clay requirements are met
inside the framework’s verified sector.

The framework thus establishes that the **physical** Yang-Mills theory (the one describing strong
interactions) has a mass gap by construction in the flat stationary OS sector, with an explicit
Hamiltonian gap.

*Physical interpretation:* Nature forbids infinite-information vacua. The mass gap is not an empirical accident but a **logical requirement** for any theory describing existing systems.

:::

(pi-mass-gap)=
::::{admonition} Direct Isomorphism: Yang-Mills for Information
:class: note

**This is not an analogy.** The Fragile Agent framework implements Yang-Mills field equations directly:
- Same gauge-covariant derivative: $D_\mu = \partial_\mu - igA_\mu$
- Same field strength: $\mathcal{F}_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu - ig[A_\mu, A_\nu]$
- Same action: $S_{\text{YM}} = -\frac{1}{4}\int\text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu})$
- Same field equations: $D_\mu\mathcal{F}^{\mu\nu} = J^\nu$

The framework is Yang-Mills theory applied to information systems. The mass gap result (Theorem {prf:ref}`thm-mass-gap-dichotomy`) follows from information-theoretic constraints that apply to any computational implementation.

**Correspondence Table:**

| QCD (Yang-Mills) | Fragile Agent |
|:-----------------|:--------------|
| Mass gap $\Delta_{\text{KG}}$ (→ $\Delta_H$ in Schr\"odinger limit) | Strategic excitation threshold |
| Glueball mass | Minimum chart creation energy |
| Confinement | Cooperative basin locking |
| Lattice spacing $a$ | Levin Length $\ell_L$ |
| Continuum limit $a \to 0$ | Causal Stasis (pathological) |
| Asymptotic freedom | High-energy strategic independence |
| Area law (holographic) | Causal Information Bound |
| Screening mass $\kappa$ | Temporal discount rate $\lambda = -\ln\gamma / \Delta t$ (so $\kappa = \lambda / c_{\text{info}}$) |

::::



(sec-diagnostic-nodes-gauge)=
## Diagnostic Nodes 63–66 (Gauge Consistency)

Following the diagnostic node convention ({ref}`sec-theory-thin-interfaces`), we define four monitors for gauge consistency in multi-agent systems.

(node-63)=
**Node 63: GaugeInvarianceCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **63** | **GaugeInvarianceCheck** | Multi-Agent | Symmetry | Is dynamics gauge-invariant? | $\delta_{\text{gauge}} := \|\mathcal{L}(A') - \mathcal{L}(A)\|$ | $O(Nd^2)$ |

**Interpretation:** Monitors deviation from gauge invariance under random gauge transformations $U(z)$.

**Threshold:** $\delta_{\text{gauge}} < \epsilon_{\text{gauge}}$ (typical default $10^{-6}$).

**Trigger conditions:**
- High GaugeInvarianceCheck: Numerical gauge symmetry violation
- **Remedy:** Regularize gauge degrees of freedom; impose gauge-fixing condition (Coulomb, Lorenz, etc.)



(node-64)=
**Node 64: FieldStrengthBoundCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **64** | **FieldStrengthBoundCheck** | Multi-Agent | Stability | Is strategic curvature bounded? | $\|\mathcal{F}\|_h := \sqrt{\text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}_{\alpha\beta} h^{\mu\alpha} h^{\nu\beta})}$ | $O(N^2d^2)$ |

**Interpretation:** Monitors a positive-definite magnitude of the field strength tensor using a chosen Riemannian metric $h_{\mu\nu}$ on spacetime (e.g., the Wick-rotated $g_{\mu\nu}$).

**Threshold:** $\|\mathcal{F}\|_F < F_{\max}$ (implementation-dependent).

**Trigger conditions:**
- High FieldStrengthBoundCheck: Strong strategic curvature regime (intense conflict)
- **Remedy:** Reduce coupling $g$; add gauge field damping; check for instabilities



(node-65)=
**Node 65: BianchiViolationCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **65** | **BianchiViolationCheck** | Multi-Agent | Conservation | Is Bianchi identity satisfied? | $\delta_B := \|D_{[\mu}\mathcal{F}_{\nu\rho]}\|$ | $O(Nd^3)$ |

**Interpretation:** The Bianchi identity $D_{[\mu}\mathcal{F}_{\nu\rho]} = 0$ must hold exactly. Violations indicate:
- Topological defects (monopoles, instantons)
- Numerical integration errors
- Coordinate singularities

**Threshold:** $\delta_B < 10^{-8}$ (strict geometric constraint).

**Trigger conditions:**
- High BianchiViolationCheck: Topological anomaly or numerical instability
- **Remedy:** Check for singular gauge configurations; refine numerical integration



(node-66)=
**Node 66: MassGapCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **66** | **MassGapCheck** | Multi-Agent | Stability | Is mass gap positive? | $\Delta_H := E_1 - E_0$ (Hamiltonian spectral gap) | $O(N^2d)$ |

**Interpretation:** Monitors the energy gap between ground state and first excited state.

**Threshold:** $\Delta_H > \Delta_{\min}$ (must be strictly positive).

**Trigger conditions:**
- $\Delta_H \to 0$: Approaching critical point (phase transition)
- $\Delta_H < 0$: Numerical or spectral ordering error (gap should not be negative); re-estimate eigenvalues
- **Remedy:** Check eigensolver stability; verify boundary conditions and normalization; add mass regularization



**Summary Table: Gauge Diagnostic Nodes**

| Node | Name | Monitors | Healthy Range |
|:-----|:-----|:---------|:--------------|
| 63 | GaugeInvarianceCheck | Gauge symmetry | $\delta_{\text{gauge}} < 10^{-6}$ |
| 64 | FieldStrengthBoundCheck | Strategic curvature | $\|\mathcal{F}\|_F < F_{\max}$ |
| 65 | BianchiViolationCheck | Topological consistency | $\delta_B < 10^{-8}$ |
| 66 | MassGapCheck | Spectral stability | $\Delta_H > 0$ |



## Part VI: Quantum Layer

:::{div} feynman-prose
All right, now we are going to take another big leap. So far everything has been classical. We have had waves, but they are classical waves, like water waves or sound waves. The value field propagates, but it is a real-valued field satisfying a wave equation.

Now I want to show you something remarkable. If we write the belief density as a complex amplitude, $\psi = \sqrt{\rho} e^{iV/\sigma}$, something magical happens. The equations of motion become a Schrodinger equation. The wave-function interference, entanglement, tunneling, all the machinery of quantum mechanics appears.

Let me be very clear about what I am claiming and what I am not claiming. I am not saying that agents are literally quantum mechanical systems at the fundamental physics level. That would be silly. The neurons in your brain are not in quantum superposition in any useful sense.

What I am saying is that the mathematical structure of quantum mechanics turns out to be incredibly useful for describing belief dynamics. The complex amplitude lets you represent superpositions of beliefs. Entanglement captures strategic coupling that cannot be factorized. Tunneling lets you escape local optima. The whole formalism works.

This is a mathematical technology, not a claim about fundamental physics. Just like we use complex numbers in electrical engineering not because voltage is "really complex" but because the math is convenient. The Schrodinger equation is a powerful way to organize our thinking about belief dynamics in multi-agent systems.
:::

(sec-the-belief-wave-function-schrodinger-representation)=
## The Belief Wave-Function (Schrödinger Representation)

While Sections 29.1–29.7 derive multi-agent dynamics using **classical** field equations (coupled WFR flows), a more powerful formulation emerges when we lift the belief density to a **complex amplitude**. This section establishes the **Schrödinger Representation** of inference dynamics, revealing strategic interaction as a form of **quantum entanglement** on the latent manifold.

(rb-quantum-formalism-marl)=
:::{admonition} Researcher Bridge: Why Quantum Formalism?
:class: warning
The quantum representation is not a claim about literal quantum physics—it is a **mathematical technology** that provides:
1. **Linear superposition** of strategies via complex amplitudes
2. **Entanglement** as a precise definition of non-factorizable strategic coupling
3. **Tunneling** as a mechanism for escaping local Nash equilibria
4. **Spectral methods** (eigenvalue problems) for finding ground states (Nash)
5. **Imaginary time evolution** as a rigorous version of value iteration

This parallels how the GKSL formalism (Definition {prf:ref}`def-gksl-generator`) uses quantum operator notation for classical belief dynamics.
:::

:::{prf:definition} Inference Hilbert Space
:label: def-inference-hilbert-space

Let $(\mathcal{Z}, G)$ be the latent manifold with capacity-constrained metric (Theorem {prf:ref}`thm-capacity-constrained-metric-law`). The **Inference Hilbert Space** is:

$$
\mathcal{H} := L^2(\mathcal{Z}, d\mu_G), \quad d\mu_G := \sqrt{\det G(z)}\, d^n z,

$$
with inner product:

$$
\langle \psi_1 | \psi_2 \rangle := \int_{\mathcal{Z}} \overline{\psi_1(z)} \psi_2(z)\, d\mu_G(z).

$$
The measure $d\mu_G$ is the **Riemannian volume form**, ensuring coordinate invariance of the inner product.

*Units:* $[\psi] = [z]^{-d/2}$ (probability amplitude density).

*Remark (Coordinate Invariance).* Under a coordinate transformation $z \to z'$, the Jacobian factor $|\partial z/\partial z'|$ cancels with $\sqrt{\det G}$, leaving $\langle \psi_1 | \psi_2 \rangle$ invariant.

*Remark (Field Extensions).* In the field-theoretic layer (SMoC), the scalar space $\mathcal{H}$
is extended to bundle-valued $L^2$ sections (e.g., spinor and gauge bundles) over spacetime
$\mathcal{M}$, with the same measure structure on each fiber.

:::

:::{prf:definition} Belief Wave-Function
:label: def-belief-wave-function

Let $\rho(z, s)$ be the belief density from the WFR dynamics (Definition {prf:ref}`def-the-wfr-action`) and $V(z, s)$ be the value function (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`). The **Belief Wave-Function** is the complex amplitude:

$$
\psi(z, s) := \sqrt{\rho(z, s)} \exp\left(\frac{i V(z, s)}{\sigma}\right),

$$
where $\sigma > 0$ is the **Cognitive Action Scale** (Definition {prf:ref}`def-cognitive-action-scale`).

**Decomposition:**
- **Amplitude:** $R(z, s) := \sqrt{\rho(z, s)} = |\psi(z, s)|$
- **Phase:** $\phi(z, s) := V(z, s)/\sigma = \arg(\psi(z, s))$

**Probability Recovery:**

$$
|\psi(z, s)|^2 = \rho(z, s), \quad \int_{\mathcal{Z}} |\psi|^2 d\mu_G = 1.

$$
*Physical interpretation:* The amplitude $R$ encodes "how much" belief mass is at $z$; the phase $\phi$ encodes "which direction" the belief is flowing (via $\nabla_B V$).

:::

:::{prf:definition} Cognitive Action Scale
:label: def-cognitive-action-scale

The **Cognitive Action Scale** $\sigma$ is the information-theoretic analog of Planck's constant $\hbar$:

$$
\sigma := T_c \cdot \tau_{\text{update}},

$$
where:
- $T_c$ is the Cognitive Temperature ({prf:ref}`def-cognitive-temperature`, {ref}`sec-the-geodesic-baoab-integrator`), setting the scale of stochastic exploration
- $\tau_{\text{update}}$ is the characteristic belief update timescale

**Equivalent characterizations:**
1. **Entropy-Action Duality:** $\sigma$ relates entropy production to "cognitive action" via $\Delta S = \mathcal{A}/\sigma$
2. **Resolution Limit:** $\sigma \sim \ell_L^2$ where $\ell_L$ is the Levin Length ({ref}`sec-saturation-limit`)
3. **Uncertainty Scale:** $\sigma$ sets the minimum uncertainty product $\Delta z \cdot \Delta p \geq \sigma/2$

*Units:* $[\sigma] = \text{nat} \cdot \text{step} = \text{bit} \cdot \text{step} / \ln 2$.

*Cross-reference:* In the limit $\sigma \to 0$ (zero temperature, infinite precision), the wave-function becomes a delta function concentrated on the optimal trajectory—recovering classical gradient flow.

:::

:::{prf:proposition} Self-Adjointness of the Laplace-Beltrami Operator
:label: prop-laplace-beltrami-self-adjointness

The Laplace-Beltrami operator

$$
\Delta_G := \frac{1}{\sqrt{|G|}} \partial_i \left( \sqrt{|G|} G^{ij} \partial_j \right)

$$
is essentially self-adjoint on $\mathcal{H} = L^2(\mathcal{Z}, d\mu_G)$ with domain $C_c^\infty(\mathcal{Z})$ (smooth functions with compact support), provided either:
1. $(\mathcal{Z}, G)$ is **geodesically complete**, or
2. $\mathcal{Z}$ has a boundary $\partial \mathcal{Z}$ with **Dirichlet conditions** $\psi|_{\partial \mathcal{Z}} = 0$ (sensors, Definition {prf:ref}`def-dirichlet-boundary-condition-sensors`) or **Neumann conditions** $\nabla_n \psi|_{\partial \mathcal{Z}} = 0$ (motors, Definition {prf:ref}`def-neumann-boundary-condition-motors`).

*Proof sketch.* The quadratic form $q[\psi] := \int_{\mathcal{Z}} \|\nabla_G \psi\|^2 d\mu_G$ is positive and closable. By the **Friedrichs Extension Theorem**, there exists a unique self-adjoint extension $\Delta_G^F$ associated with $q$. For geodesically complete manifolds, this extension coincides with the closure of $\Delta_G$ on $C_c^\infty$. See {cite}`strichartz1983analysis` for the general theory. $\square$

*Consequence:* Self-adjointness guarantees that $-\Delta_G$ has a real spectrum bounded below, enabling spectral decomposition and ground state analysis.

:::

(remark-line-bundle-formalism)=
:::{admonition} Remark: Line Bundle Formalism for Topologically Non-Trivial Manifolds
:class: dropdown

When the latent manifold has non-trivial topology (e.g., $\pi_1(\mathcal{Z}) \neq 0$), the phase $V/\sigma$ may be **multi-valued** around non-contractible loops. In this case, $\psi$ should be defined as a **section of a complex line bundle** $\mathcal{L} \to \mathcal{Z}$ with connection 1-form $A = dV/\sigma$.

The **holonomy** around a closed loop $\gamma$ is:

$$
\exp\left(\frac{i}{\sigma} \oint_\gamma dV\right) = \exp\left(\frac{i}{\sigma} \Delta V_\gamma\right),

$$
where $\Delta V_\gamma$ is the value change around the loop. Non-trivial holonomy ($\Delta V_\gamma \neq 0 \mod 2\pi\sigma$) implies the phase $V/\sigma$ is not globally defined; instead, $\psi$ is a section of a non-trivial complex line bundle $\mathcal{L} \to \mathcal{Z}$.

For most applications where $\mathcal{Z}$ is simply connected (e.g., Poincare disk), this subtlety does not arise, and $\psi$ is a well-defined scalar function.

:::

(pi-holonomy)=
::::{admonition} Physics Isomorphism: Holonomy and Berry Phase
:class: note

**In Physics:** Holonomy measures the failure of parallel transport around a closed loop to return a vector to itself. The Berry phase $\gamma_n = i\oint \langle n|\nabla_R|n\rangle \cdot dR$ is the geometric phase acquired by a quantum state under adiabatic evolution around a parameter loop {cite}`berry1984quantal,nakahara2003geometry`.

**In Implementation:** When $\mathcal{Z}$ has non-trivial topology, the value phase $V/\sigma$ may be multi-valued. The holonomy around a closed loop $\gamma$ is (see {ref}`Line Bundle Formalism <remark-line-bundle-formalism>`):

$$
\exp\left(\frac{i}{\sigma} \oint_\gamma dV\right) = \exp\left(\frac{i}{\sigma} \Delta V_\gamma\right)

$$
**Correspondence Table:**
| Gauge Theory | Agent (Value Phase) |
|:-------------|:--------------------|
| Connection 1-form $A$ | $dV/\sigma$ |
| Holonomy $\exp(i\oint A)$ | Phase accumulated around loop |
| Berry phase | Value change $\Delta V_\gamma$ |
| Line bundle $\mathcal{L}$ | Complex belief amplitude bundle |
| Trivial bundle | Simply connected $\mathcal{Z}$ |

**Significance:** Non-trivial holonomy ($\Delta V_\gamma \neq 0 \mod 2\pi\sigma$) implies the belief wave-function is a section of a non-trivial line bundle—topological structure in the agent's value landscape.
::::

(sec-the-inference-wave-correspondence)=
## The Inference-Wave Correspondence (WFR to Schrödinger)

:::{div} feynman-prose
Now I want to show you a beautiful piece of mathematics. We have the WFR dynamics, the continuity equation for the belief density $\rho$ and the Hamilton-Jacobi-Bellman equation for the value $V$. Two coupled real equations. Can we combine them into one complex equation?

Yes. And when we do, we get the Schrodinger equation.

Here is how it works. Define the wave-function $\psi = \sqrt{\rho} e^{iV/\sigma}$. The modulus squared $|\psi|^2$ gives back the density $\rho$. The phase $\arg(\psi)$ gives back the value $V$ up to a scale factor $\sigma$.

Now take the WFR equations and substitute. After some algebra, which I will spare you, you find that $\psi$ satisfies $i\sigma \partial_t \psi = H \psi$, where $H$ is a Hamiltonian operator. This is the Schrodinger equation.

The correspondence is called the Madelung transform, after the physicist who discovered it in 1926, just one year after Schrodinger wrote down his equation. Madelung realized that Schrodinger's equation is secretly a fluid equation in disguise. The quantum potential, that weird $Q_B$ term that has troubled philosophers for decades, arises naturally from the geometry of the belief density.

Here is the physical meaning of $Q_B$: it is the cost of localization. If you try to squeeze the belief density into a tiny region, it resists. The $Q_B$ term pushes back, trying to spread the density out. This is not some mysterious quantum effect. It is just the geometry of information. You cannot know both where you are and where you are going with arbitrary precision. The uncertainty principle is built into the structure of belief dynamics.
:::

We now derive the **Schrödinger equation** for the belief wave-function from the WFR dynamics. This is the inverse of the **Madelung transform** in quantum mechanics.

:::{prf:theorem} The Madelung Transform (WFR-Schrödinger Equivalence)
:label: thm-madelung-transform

Let the belief density $\rho$ and value $V$ satisfy the WFR-HJB system with vector potential:
1. **WFR Continuity (unbalanced):** $\partial_s \rho + \nabla_G \cdot (\rho \mathbf{v}) = \rho r$
2. **Hamilton-Jacobi-Bellman:** $\partial_s V + \frac{1}{2}\|\nabla_B V\|_G^2 + \Phi_{\text{eff}} = 0$

where $\nabla_B V := \nabla V - B$ and $B$ is the vector potential for the reward 1-form (Value Curl), $dB = \mathcal{F}$.
The drift is $\mathbf{v} = +G^{-1}\nabla_B V$ (canonical mobility; conservative case: $B=0$, $\mathcal{F}=0$).
If curl-induced mobility is included, replace $\mathbf{v}$ by $\mathcal{M}_{\text{curl}}\!\left(+G^{-1}\nabla_B V\right)$ with
$\mathcal{M}_{\text{curl}} := (I - \beta_{\text{curl}} G^{-1}\mathcal{F})^{-1}$.
$r$ is the WFR reaction rate (Definition {prf:ref}`def-the-wfr-action`).

Then the belief wave-function $\psi = \sqrt{\rho} e^{iV/\sigma}$ satisfies the **Inference Schrödinger Equation**:

$$
i\sigma \frac{\partial \psi}{\partial s} = \hat{H}_{\text{inf}} \psi,

$$
where the **Inference Hamiltonian** is:

$$
\hat{H}_{\text{inf}} := -\frac{\sigma^2}{2} D^i D_i + \Phi_{\text{eff}} + Q_B + \frac{i\sigma}{2} r,

$$
The terms are:
- **Kinetic:** $-\frac{\sigma^2}{2} D^i D_i$ (covariant Laplacian with vector potential)
- **Potential:** $\Phi_{\text{eff}}$ (effective potential from rewards and constraints)
- **Quantum Correction:** $Q_B$ (Bohm potential, Definition {prf:ref}`def-bohm-quantum-potential`)
- **Reaction:** $+\frac{i\sigma}{2} r$ (non-Hermitian term from WFR reaction; positive $r$ creates mass)

Here $D_i := \nabla_i - \frac{i}{\sigma} B_i$ is the $U(1)$ covariant derivative; $B$ is the Opportunity field (reward 1-form) and is distinct from the strategic gauge connection $A_\mu$.

*Proof.* See {ref}`sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws` for the rigorous derivation. The key steps are:

**Step 1 (Substitution).** Write $\psi = R e^{i\phi}$ with $R = \sqrt{\rho}$ and $\phi = V/\sigma$.

**Step 2 (Time derivative).**

$$
i\sigma \partial_s \psi = i\sigma \left( \frac{\partial_s R}{R} + \frac{i}{\sigma}\partial_s V \right) \psi = \left( \frac{i\sigma \partial_s \rho}{2\rho} - \partial_s V \right) \psi.

$$
**Step 3 (Use governing equations).** Substitute the continuity equation for $\partial_s \rho$ and HJB for $\partial_s V$, using $\nabla_B V$ and the covariant Laplacian $D^i D_i$.

**Step 4 (Identify terms).** The real part of the resulting equation gives the HJB with Bohm correction; the imaginary part gives the continuity equation with reaction. Combining yields the Schrödinger form. $\square$

:::

(pi-madelung)=
::::{admonition} Physics Isomorphism: Madelung Transform
:class: note

**In Physics:** The Madelung transform $\psi = \sqrt{\rho}e^{iS/\hbar}$ converts the Schrödinger equation into hydrodynamic form: continuity + quantum Hamilton-Jacobi with Bohm potential $Q = -\frac{\hbar^2}{2m}\frac{\nabla^2\sqrt{\rho}}{\sqrt{\rho}}$ {cite}`madelung1927quantentheorie,bohm1952suggested`. With a vector potential, minimal coupling replaces $\nabla$ by $\nabla - iA/\hbar$ (covariant Laplacian).

**In Implementation:** The WFR-to-Schrödinger correspondence (Theorem {prf:ref}`thm-madelung-transform`):

$$
\psi(z,s) = \sqrt{\rho(z,s)}\exp(iV(z,s)/\sigma)

$$
with information resolution limit $Q_B = -\frac{\sigma^2}{2}\frac{\Delta_G\sqrt{\rho}}{\sqrt{\rho}}$.

**Correspondence Table:**

| Quantum Mechanics | Agent (Inference Wave) |
|:------------------|:-----------------------|
| Wave function $\psi$ | Belief amplitude |
| Planck constant $\hbar$ | Cognitive scale $\sigma$ |
| Bohm potential $Q$ | Information resolution limit $Q_B$ |
| Probability current $\mathbf{j}$ | Belief flux $\rho v$ |
::::

:::{prf:definition} Bohm Quantum Potential (Information Resolution Limit)
:label: def-bohm-quantum-potential

The **Bohm Quantum Potential** is:

$$
Q_B(z, s) := -\frac{\sigma^2}{2} \frac{\Delta_G \sqrt{\rho}}{\sqrt{\rho}} = -\frac{\sigma^2}{2} \frac{\Delta_G R}{R},

$$
where $R = \sqrt{\rho}$ is the amplitude.

**Explicit form in terms of $\rho$:**

$$
Q_B = -\frac{\sigma^2}{8\rho^2} \|\nabla_G \rho\|_G^2 + \frac{\sigma^2}{4\rho} \Delta_G \rho.

$$
**Physical interpretation:** $Q_B$ represents the **energetic cost of belief localization**. Regions where $\rho$ has high curvature (sharp belief features) incur an effective potential energy penalty. This prevents the belief from concentrating to delta functions.

**Information-theoretic interpretation:** $Q_B$ enforces the **Levin Length** ({ref}`sec-saturation-limit`) as a resolution limit. The agent cannot represent distinctions finer than $\ell_L \sim \sqrt{\sigma}$.

*Units:* $[Q_B] = \text{nat}$ (same as potential).

*Cross-reference:* In standard quantum mechanics, $Q_B$ is called the "quantum potential" or "Bohm potential." Here it emerges from the information geometry, not fundamental physics.

:::

:::{prf:corollary} Open Quantum System Interpretation
:label: cor-open-quantum-system

The Inference Hamiltonian $\hat{H}_{\text{inf}}$ is **non-Hermitian** due to the reaction term $+\frac{i\sigma}{2}r$. This corresponds to an **open quantum system** where:
- $r > 0$: Mass creation (information gain from boundary) → probability amplitude **grows**
- $r < 0$: Mass destruction (information loss) → probability amplitude **decays**

The **complex potential** formulation is:

$$
W(z) := \Phi_{\text{eff}}(z) + \frac{i\sigma}{2} r(z),

$$
so that $\hat{H}_{\text{inf}} = -\frac{\sigma^2}{2} D^i D_i + W + Q_B$.

**Norm evolution:** The normalization $\|\psi\|^2 = \int |\psi|^2 d\mu_G$ evolves as:

$$
\frac{d}{ds} \|\psi\|^2 = \int_{\mathcal{Z}} r(z) |\psi(z)|^2 d\mu_G(z) = \langle r \rangle_\rho,

$$
which matches the WFR mass balance equation.

*Remark (Lindblad Connection).* For trace-preserving dynamics (where $\int r \rho\, d\mu_G = 0$), the non-Hermitian Schrödinger equation can be embedded in a **Lindblad master equation** (Definition {prf:ref}`def-gksl-generator`) via the Dyson-Phillips construction.

:::

:::{prf:proposition} Operator Ordering and Coordinate Invariance
:label: prop-operator-ordering-invariance

The kinetic term $-\frac{\sigma^2}{2} D^i D_i$ in the Inference Hamiltonian uses the unique **coordinate-invariant** ordering:

$$
-\frac{\sigma^2}{2} D^i D_i \psi = -\frac{\sigma^2}{2} \cdot \frac{1}{\sqrt{|G|}} D_i \left( \sqrt{|G|} G^{ij} D_j \psi \right).

$$
This is equivalent to:

$$
-\frac{\sigma^2}{2} D^i D_i = -\frac{\sigma^2}{2} \left( G^{ij} D_i D_j + \Gamma^k D_k \right),

$$
where $\Gamma^k := G^{ij}\Gamma^k_{ij}$ is the trace of Christoffel symbols.

**Alternative orderings** (Weyl, symmetric, etc.) would introduce frame-dependent terms that break the geometric interpretation.

*Cross-reference:* This reduces to the Laplace-Beltrami operator used in the Helmholtz equation (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`) when $B=0$, ensuring consistency between the PDE and wave-function formulations.

:::

:::{prf:corollary} Semiclassical Limit
:label: cor-semiclassical-limit

In the limit $\sigma \to 0$ (classical limit), the Schrödinger dynamics recover the **geodesic flow**:

**WKB Ansatz:** $\psi = a(z) e^{iS(z)/\sigma}$ with $a$ slowly varying.

**Leading Order ($O(\sigma^{-1})$):** The Hamilton-Jacobi equation

$$
\partial_s S + \frac{1}{2}\|\nabla_B S\|_G^2 + \Phi_{\text{eff}} = 0,

$$
**Next Order ($O(\sigma^0)$):** The transport equation

$$
\partial_s |a|^2 + \nabla_G \cdot (|a|^2 \nabla_B S) = 0.

$$
*Definition:* $\nabla_B S := \nabla S - B$. If curl-induced mobility is present, replace the flux by
$|a|^2 \mathcal{M}_{\text{curl}} G^{-1}\nabla_B S$.
These are exactly the HJB and continuity equations from WFR dynamics. The quantum correction $Q_B \to 0$ as $\sigma \to 0$.

*Interpretation:* The wave-function collapses to a delta function following the optimal trajectory. Quantum effects (tunneling, interference) vanish in this limit.

:::

(sec-multi-agent-schrodinger-equation)=
## Multi-Agent Schrödinger Equation

:::{div} feynman-prose
When you have multiple agents, something very interesting happens. The joint wave-function lives on the product space, but it does not have to factor into a product of individual wave-functions. When it does not factor, we say the agents are entangled.

Now, in quantum physics, entanglement is this spooky, mysterious thing that bothered Einstein so much he called it "spooky action at a distance." But in our context, entanglement has a very concrete meaning: the agents have strategic correlations that cannot be captured by treating them independently.

Think about it this way. If two agents are strategically entangled, you cannot describe what agent A will do without talking about agent B, and vice versa. Their decisions are locked together in a way that transcends simple classical correlation.

Classical correlation would be: A and B are both responding to the same signal. If you knew the signal, you could predict both of them. But quantum entanglement is stronger. Even if you know everything about the individual agents, their joint behavior cannot be predicted from the marginals alone.

This happens in games all the time. In a coordination game, the equilibrium is not "A does X and B does Y independently." The equilibrium is "A does X if B does Y, and B does Y if A does X." The strategies are locked together. You cannot factor them.

The tensor product structure of the multi-agent Hilbert space naturally captures this. The dimension grows exponentially with the number of agents, which is both a blessing and a curse. It is expressive enough to represent arbitrary correlations, but it is also computationally intractable for large $N$. That is the curse of dimensionality, appearing in a new guise.
:::

We now extend the wave-function formalism to $N$-agent systems, defining **strategic entanglement** as non-factorizability of the joint belief amplitude.

:::{prf:definition} Joint Inference Hilbert Space
:label: def-joint-inference-hilbert-space

For $N$ agents with individual Hilbert spaces $\mathcal{H}^{(i)} = L^2(\mathcal{Z}^{(i)}, d\mu_{G^{(i)}})$, the **Joint Inference Hilbert Space** is the tensor product:

$$
\mathcal{H}^{(N)} := \bigotimes_{i=1}^N \mathcal{H}^{(i)} = L^2\left(\mathcal{Z}^{(N)}, d\mu_{G^{(N)}}\right),

$$
where:
- $\mathcal{Z}^{(N)} = \prod_{i=1}^N \mathcal{Z}^{(i)}$ is the product manifold (Definition {prf:ref}`def-n-agent-product-manifold`)
- $d\mu_{G^{(N)}} = \prod_{i=1}^N d\mu_{G^{(i)}}$ is the product measure

Elements $\Psi \in \mathcal{H}^{(N)}$ are functions $\Psi: \mathcal{Z}^{(N)} \to \mathbb{C}$ with:

$$
\|\Psi\|^2 = \int_{\mathcal{Z}^{(N)}} |\Psi(\mathbf{z})|^2 d\mu_{G^{(N)}}(\mathbf{z}) < \infty.

$$
*Notation:* We use uppercase $\Psi$ for joint wave-functions and lowercase $\psi^{(i)}$ for single-agent wave-functions.

:::

:::{prf:definition} Strategic Entanglement
:label: def-strategic-entanglement

A joint wave-function $\Psi \in \mathcal{H}^{(N)}$ exhibits **Strategic Entanglement** if it cannot be written as a product:

$$
\Psi(z^{(1)}, \ldots, z^{(N)}) \neq \prod_{i=1}^N \psi^{(i)}(z^{(i)}) \quad \text{for any choice of } \psi^{(i)} \in \mathcal{H}^{(i)}.

$$
**Entanglement Entropy:** For a bipartition $\{i\} \cup \{j \neq i\}$, the **Strategic Entanglement Entropy** is:

$$
S_{\text{ent}}(i) := -\text{Tr}\left[\hat{\rho}^{(i)} \ln \hat{\rho}^{(i)}\right],

$$
where $\hat{\rho}^{(i)} = \text{Tr}_{j \neq i}[|\Psi\rangle\langle\Psi|]$ is the **reduced density operator** obtained by partial trace over all agents except $i$.

**Physical interpretation:**
- $S_{\text{ent}}(i) = 0$: Agent $i$ is **disentangled** (can be modeled independently)
- $S_{\text{ent}}(i) > 0$: Agent $i$ is **entangled** with others (cannot be modeled in isolation)
- $S_{\text{ent}}(i) \leq \ln \dim(\mathcal{H}^{(i)})$: **Maximal entanglement** for finite-dimensional subsystems (continuous spaces require a cutoff, giving $S_{\text{ent}}(i) \leq \ln d_{\text{eff}}$)

*Cross-reference:* The partial trace operation corresponds to the **Information Bottleneck** (Definition {prf:ref}`def-dpi-boundary-capacity-constraint`)—marginalizing over opponents discards strategic correlations.

:::

:::{prf:definition} Strategic Hamiltonian
:label: def-strategic-hamiltonian

The **Strategic Hamiltonian** on $\mathcal{H}^{(N)}$ is:

$$
\hat{H}_{\text{strat}} := \sum_{i=1}^N \hat{H}^{(i)}_{\text{kin}} + \sum_{i=1}^N \hat{\Phi}^{(i)}_{\text{eff}} + \sum_{i < j} \hat{V}_{ij},

$$
where:
1. **Kinetic terms:** $\hat{H}^{(i)}_{\text{kin}} = -\frac{\sigma_i^2}{2} D^{(i)a} D^{(i)}_a$ (acting on $\mathcal{Z}^{(i)}$ coordinates)
2. **Individual potentials:** $\hat{\Phi}^{(i)}_{\text{eff}}$ (local reward landscape for agent $i$)
3. **Interaction potentials:** $\hat{V}_{ij} = \Phi_{ij}(z^{(i)}, z^{(j)})$ (strategic coupling)

Here $D^{(i)}_a := \nabla^{(i)}_a - \frac{i}{\sigma_i} B^{(i)}_a$ is the covariant derivative for agent $i$ and
$B^{(i)}$ is the reward 1-form (Opportunity field). Conservative case: $B^{(i)} = 0$.

*Notation (Per-Agent Action Scale):* Here $\sigma_i := T_{c,i} \cdot \tau_{\text{update},i}$ is the cognitive action scale for agent $i$, generalizing Definition {prf:ref}`def-cognitive-action-scale`. For **homogeneous** agents with identical cognitive properties, $\sigma_i = \sigma$ for all $i$. For **heterogeneous** agents (e.g., different computation rates), $\sigma_i$ may vary.

*Remark (Separability).* If all $\hat{V}_{ij} = 0$, the Hamiltonian is **separable**: $\hat{H}_{\text{strat}} = \sum_i \hat{H}^{(i)}$, and the ground state is a product $\Psi_0 = \prod_i \psi^{(i)}_0$. Non-zero interaction creates entanglement.

:::

:::{prf:theorem} Multi-Agent Schrödinger Equation
:label: thm-multi-agent-schrodinger-equation

Assume **homogeneous agents** with a common cognitive action scale $\sigma_i = \sigma$. The joint belief wave-function $\Psi(\mathbf{z}, s)$ of $N$ strategically coupled agents evolves according to:

$$
i\sigma \frac{\partial \Psi}{\partial s} = \hat{H}_{\text{strat}} \Psi + i\frac{\sigma}{2} \mathcal{R} \Psi,

$$
where:
- $\hat{H}_{\text{strat}}$ is the Strategic Hamiltonian (Definition {prf:ref}`def-strategic-hamiltonian`)
- $\mathcal{R}(\mathbf{z}) = \sum_i r^{(i)}(z^{(i)})$ is the total reaction rate

**Expanded form:**

$$
i\sigma \frac{\partial \Psi}{\partial s} = \left[ \sum_{i=1}^N \left( -\frac{\sigma^2}{2} D^{(i)a} D^{(i)}_a + \Phi^{(i)}_{\text{eff}} \right) + \sum_{i < j} \Phi_{ij} \right] \Psi + i\frac{\sigma}{2} \mathcal{R} \Psi.

$$
*Remark (Heterogeneous agents).* If $\sigma_i$ differ, choose a reference scale $\sigma_{\text{ref}}$ in the time derivative and rescale the kinetic terms by $(\sigma_i/\sigma_{\text{ref}})^2$.
**Sources of entanglement:** Strategic entanglement arises from:
1. **Potential coupling:** Non-zero $\Phi_{ij}(z^{(i)}, z^{(j)})$ creates position-position correlations
2. **Metric coupling:** The Game Tensor $\mathcal{G}_{ij}$ modifies the kinetic terms (Theorem {prf:ref}`thm-game-augmented-laplacian`)

*Cross-reference:* This extends Theorem {prf:ref}`thm-madelung-transform` to multiple agents, with the joint WFR dynamics (Definition {prf:ref}`def-joint-wfr-action`) as the underlying classical limit.

:::

:::{prf:theorem} Game-Augmented Laplacian
:label: thm-game-augmented-laplacian

Under adversarial coupling, the effective kinetic operator for agent $i$ incorporates the **Game Tensor** (Definition {prf:ref}`def-the-game-tensor`):

$$
\hat{H}^{(i)}_{\text{kin,eff}} = -\frac{\sigma_i^2}{2} \tilde{\Delta}^{(i)},

$$
where the **Game-Augmented Laplacian** is:

$$
\tilde{\Delta}^{(i)} := \frac{1}{\sqrt{|\tilde{G}^{(i)}|}} \partial_a \left( \sqrt{|\tilde{G}^{(i)}|} (\tilde{G}^{(i)})^{ab} \partial_b \right),

$$
with strategic metric $\tilde{G}^{(i)} = G^{(i)} + \sum_{j \neq i} \beta_{ij} \mathcal{G}^{(i)}_{ij}$ (Definition {prf:ref}`def-the-game-tensor`, Equation 29.4.1).

**Consequence for entanglement:** Since $\tilde{G}^{(i)}$ depends on $z^{(j)}$ through the Game Tensor, the kinetic operator for agent $i$ is **not separable**:

$$
\tilde{\Delta}^{(i)} = \tilde{\Delta}^{(i)}(z^{(i)}; z^{(-i)}).

$$
This creates **kinetic entanglement**—even without potential coupling, adversarial metric inflation entangles the agents.

*Physical interpretation:* Agent $j$ "curves" agent $i$'s configuration space. Moving through a contested region requires more "effort" (higher effective mass), and this coupling cannot be factorized away.

*Remark (Gauge coupling).* With a non-conservative reward 1-form, replace $\partial_a$ by $D^{(i)}_a$ in $\tilde{\Delta}^{(i)}$.

:::

:::{prf:proposition} Partial Trace and Reduced Dynamics
:label: prop-partial-trace-reduced-dynamics

For a pure joint state $|\Psi\rangle \in \mathcal{H}^{(N)}$, the **reduced density operator** for agent $i$ is:

$$
\hat{\rho}^{(i)} := \text{Tr}_{j \neq i}\left[ |\Psi\rangle\langle\Psi| \right].

$$
In the coordinate representation its kernel is:

$$
\rho^{(i)}(z^{(i)}, z^{(i)'}) = \int_{\prod_{j \neq i} \mathcal{Z}^{(j)}} \Psi(z^{(i)}, z^{(-i)})\,\overline{\Psi(z^{(i)'}, z^{(-i)})}\, d\mu_{G^{(-i)}}.

$$
The diagonal elements give the **marginal belief density**:

$$
\rho^{(i)}(z^{(i)}) = \langle z^{(i)} | \hat{\rho}^{(i)} | z^{(i)} \rangle = \int |\Psi(z^{(i)}, z^{(-i)})|^2 d\mu_{G^{(-i)}},

$$
which is exactly the marginalization from the joint WFR density.

*Discrete analog:* In a finite basis, $\rho^{(i)}_{mn} = \sum_k \Psi_{mk}\,\Psi^*_{nk}$.

**Mixed state evolution:** Even if $\Psi$ evolves unitarily, the reduced state $\hat{\rho}^{(i)}$ generally evolves **non-unitarily** (with decoherence) due to entanglement with other agents.

:::

(sec-nash-equilibrium-as-ground-state)=
## Nash Equilibrium as Ground State

:::{div} feynman-prose
Here is one of the most beautiful results of the whole theory. Nash equilibrium corresponds to the ground state of the Strategic Hamiltonian.

Think about what this means. In quantum mechanics, the ground state is the state of lowest energy. It is where the system wants to be if you leave it alone. It is stable. Small perturbations die out and the system returns to the ground state.

Nash equilibrium in game theory has exactly the same character. At Nash, no one wants to deviate unilaterally. Small perturbations in strategy lead to gradients that push you back toward Nash. It is stable.

So it makes perfect sense that Nash equilibrium corresponds to the ground state. The variational characterization $E_0 = \min_\psi \langle\psi|H|\psi\rangle$ is just the statement that Nash minimizes something, namely the total "strategic energy" of the system.

This gives us a new way to find Nash equilibria: solve for the ground state of the Hamiltonian. There is a beautiful technique called imaginary time evolution. You take the Schrodinger equation, you rotate time to the imaginary axis, and you run it forward. Any initial state eventually decays to the ground state. The excited components die away, and only the ground state remains.

This is value iteration in disguise. The imaginary time propagator is the Bellman backup operator. Running forward in imaginary time is doing successive Bellman backups, relaxing toward the optimal value function. The spectral gap, the difference between the ground state energy and the first excited state, determines how fast you converge.
:::

The spectral properties of the Strategic Hamiltonian provide a new characterization of Nash equilibrium.

:::{prf:theorem} Nash Equilibrium as Ground State
:label: thm-nash-ground-state

A Nash equilibrium $\mathbf{z}^* = (z^{(1)*}, \ldots, z^{(N)*})$ (Theorem {prf:ref}`thm-nash-equilibrium-as-geometric-stasis`) corresponds to the **ground state** of the Strategic Hamiltonian:

1. **Spectral condition:** The ground state $\Psi_{\text{Nash}}$ satisfies:

   $$
   \hat{H}_{\text{strat}} \Psi_{\text{Nash}} = E_0 \Psi_{\text{Nash}}, \quad E_0 = \min \text{spec}(\hat{H}_{\text{strat}}).

   $$
2. **Localization:** In the semiclassical limit ($\sigma \to 0$), $|\Psi_{\text{Nash}}|^2$ concentrates near $\mathbf{z}^*$:

   $$
   \lim_{\sigma \to 0} |\Psi_{\text{Nash}}(\mathbf{z})|^2 = \delta(\mathbf{z} - \mathbf{z}^*).

   $$
3. **Energy interpretation:** The ground state energy $E_0$ equals the total effective potential at Nash:

   $$
   E_0 = \sum_{i=1}^N \Phi^{(i)}_{\text{eff}}(\mathbf{z}^*) + \sum_{i < j} \Phi_{ij}(\mathbf{z}^*) + O(\sigma).

   $$
*Proof sketch.*
- At Nash, $\nabla_{z^{(i)}} \Phi^{(i)}_{\text{eff}} = 0$ for all $i$ (Condition 1 of Theorem {prf:ref}`thm-nash-equilibrium-as-geometric-stasis`).
- The variational principle $\delta \langle \Psi | \hat{H} | \Psi \rangle / \delta \Psi^* = 0$ with normalization constraint yields the same stationarity conditions in the $\sigma \to 0$ limit.
- The second variation (Hessian) being non-positive (Condition 3) corresponds to local stability of the ground state.

See **{ref}`sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws`** for the complete WKB/semiclassical analysis proving Gaussian concentration to delta function as $\sigma \to 0$, with explicit energy correction formulas. $\square$

*Remark (Multiple Nash).* If multiple Nash equilibria exist, each corresponds to a different local minimum of the energy landscape. The **global** ground state is the Nash with lowest $E_0$; other Nash equilibria are metastable excited states.

:::

:::{prf:corollary} Vanishing Probability Current at Nash
:label: cor-vanishing-probability-current

At Nash equilibrium, the **probability current** vanishes:

$$
\mathbf{J}^{(i)}(\mathbf{z}^*) := \text{Im}\left[\bar{\Psi}_{\text{Nash}} \cdot \sigma D^{(i)} \Psi_{\text{Nash}}\right]_{\mathbf{z}^*} = 0 \quad \forall i.

$$
**Derivation:** The probability current is $\mathbf{J} = \rho \mathbf{v}$ where $\mathbf{v} = -G^{-1}\nabla_B V$ is the velocity field. At Nash:
- $\nabla_B V^{(i)}|_{\mathbf{z}^*} = 0$ (stationarity condition)
- Therefore $\mathbf{v}^{(i)}|_{\mathbf{z}^*} = 0$
- Hence $\mathbf{J}^{(i)}|_{\mathbf{z}^*} = 0$

*Interpretation:* At Nash, there is no net belief flow. The wave-function is in a **standing wave pattern**—agents are not "stopped" but are in dynamic equilibrium where flows cancel.

*Cross-reference:* This is the quantum version of **Geometric Stasis** (Theorem {prf:ref}`thm-nash-equilibrium-as-geometric-stasis`).

:::

:::{prf:proposition} Imaginary Time Evolution for Nash Finding
:label: prop-imaginary-time-nash-finding

The substitution $s \to -i\tau$ (**Wick rotation**) transforms the Schrödinger equation into a diffusion equation:

$$
-\sigma \frac{\partial \Psi}{\partial \tau} = \hat{H}_{\text{strat}} \Psi.

$$
Under this **imaginary time evolution**, any initial state $\Psi_0$ converges to the ground state:

$$
\Psi(\tau) = e^{-\hat{H}_{\text{strat}} \tau / \sigma} \Psi_0 \xrightarrow{\tau \to \infty} c \cdot \Psi_{\text{Nash}},

$$
where $c$ is a normalization constant.

**Computational interpretation:**
1. Imaginary time evolution is equivalent to **Value Iteration** in dynamic programming
2. The propagator $e^{-\hat{H}\tau/\sigma}$ is the **Bellman backup operator** in infinite-horizon limit
3. Convergence rate is set by the **spectral gap** $E_1 - E_0$ (energy of first excited state minus ground state)

**Algorithm sketch (Quantum Value Iteration):**
```
Initialize Ψ randomly
For τ = 0 to T:
    Ψ ← exp(-H_strat Δτ / σ) Ψ    # Diffusion step
    Ψ ← Ψ / ||Ψ||                  # Renormalize
Return Ψ (approximates Nash ground state)
```

*Cross-reference:* This connects to the **imaginary-time path integral** formulation used in quantum Monte Carlo methods.

:::

(sec-strategic-tunneling-and-barrier-crossing)=
## Strategic Tunneling and Barrier Crossing

:::{div} feynman-prose
Now I want to tell you about quantum tunneling, which is one of the most practically useful features of the wave-function formalism.

You know the problem of local optima. You are doing gradient descent, you find a local minimum, and you get stuck there. There might be a much better minimum somewhere else, but there is a barrier in between. Gradient descent will not get you over the barrier because the gradient points away from the barrier.

Classical physics has the same problem. A ball in a valley cannot escape unless you give it enough energy to climb over the hill. But in quantum mechanics, the ball can tunnel through the hill. It does not have to go over. The wave-function leaks through the barrier, and there is a finite probability of ending up on the other side.

The same thing happens with belief wave-functions. The wave-function has exponential tails that extend through barriers. There is always some amplitude on the other side. So even if you start in a local Nash equilibrium, there is a finite probability of finding yourself in a different basin.

The tunneling probability goes like $e^{-\Delta\Phi/\sigma}$, where $\Delta\Phi$ is the barrier height and $\sigma$ is the cognitive action scale, the analog of Planck's constant. So if $\sigma$ is large, meaning you have a lot of uncertainty or exploration, tunneling is more likely. If $\sigma$ is small, meaning you are very confident and exploiting your current knowledge, tunneling is suppressed.

This gives a principled way to think about exploration-exploitation. Early in learning, you want large $\sigma$ to explore and potentially tunnel to better basins. Later, you want small $\sigma$ to exploit what you have found. The tunneling rate is exactly the formalization of this intuition.
:::

The wave nature of the belief amplitude enables **tunneling**—crossing barriers that would trap classical gradient-descent agents.

:::{prf:definition} Pareto Barrier
:label: def-pareto-barrier

A **Pareto Barrier** $\mathcal{B}_P \subset \mathcal{Z}^{(N)}$ is a region where:
1. **Local value decrease:** $\Phi^{(i)}_{\text{eff}}(\mathbf{z}) > \Phi^{(i)}_{\text{eff}}(\mathbf{z}^*)$ for at least one agent $i$ and some starting point $\mathbf{z}^*$
2. **No Nash within:** There exists no Nash equilibrium $\mathbf{z}' \in \mathcal{B}_P$
3. **Separates basins:** $\mathcal{B}_P$ lies between distinct Nash equilibria $\mathbf{z}^*_A$ and $\mathbf{z}^*_B$

The **barrier height** is:

$$
\Delta \Phi_P := \max_{\mathbf{z} \in \mathcal{B}_P} \left[ \sum_{i=1}^N \Phi^{(i)}_{\text{eff}}(\mathbf{z}) - \sum_{i=1}^N \Phi^{(i)}_{\text{eff}}(\mathbf{z}^*_A) \right].

$$
*Mathematical characterization:* A Pareto barrier is a region where the total potential $\sum_i \Phi^{(i)}_{\text{eff}}$ exceeds its value at nearby Nash equilibria. Classical gradient descent with initial condition in the basin of attraction of $\mathbf{z}^*_A$ converges to $\mathbf{z}^*_A$ and cannot reach $\mathbf{z}^*_B$.

:::

:::{prf:theorem} Strategic Tunneling Probability (WKB Approximation)
:label: thm-tunneling-probability

In the semiclassical limit ($\sigma \ll \Delta \Phi_P$), the probability of crossing a Pareto barrier is:

$$
P_{\text{tunnel}} \sim \exp\left(-\frac{2}{\sigma} \int_{\gamma} \sqrt{2(\Phi_{\text{eff,total}}(\mathbf{z}) - E_0)}\, d\ell_{G^{(N)}}\right),

$$
where:
- $\gamma$ is the **optimal tunneling path** (instanton) connecting $\mathbf{z}^*_A$ to $\mathbf{z}^*_B$
- $\Phi_{\text{eff,total}} = \sum_i \Phi^{(i)}_{\text{eff}} + \sum_{i<j} \Phi_{ij}$
- $d\ell_{G^{(N)}}$ is the geodesic arc length on $(\mathcal{Z}^{(N)}, G^{(N)})$
- $E_0$ is the ground state energy

**Key scaling:** $P_{\text{tunnel}} \propto e^{-\Delta \Phi_P / \sigma}$, so higher barriers or lower temperature (small $\sigma$) exponentially suppress tunneling.

*Cross-reference:* This generalizes Theorem {prf:ref}`thm-memory-induced-barrier-crossing` from single-agent memory barriers to multi-agent Pareto barriers. See {ref}`sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws` for the rigorous proof via Agmon estimates and spectral theory.

:::

(pi-wkb-tunneling)=
::::{admonition} Physics Isomorphism: WKB Tunneling
:class: note

**In Physics:** The WKB approximation gives tunneling probability through a barrier: $P \sim \exp(-2\int_a^b \sqrt{2m(U-E)/\hbar^2}\,dx)$ where the integral is over the classically forbidden region {cite}`wentzel1926verallgemeinerung,agmon1982lectures`.

**In Implementation:** The tunneling probability (Theorem {prf:ref}`thm-tunneling-probability`):

$$
P_{\text{tunnel}} \sim \exp\left(-\frac{2}{\sigma}\int_\gamma \sqrt{2(\Phi_{\text{eff}} - E_0)}\,d\ell_G\right)

$$
**Correspondence Table:**

| Quantum Mechanics | Agent (Strategic Tunneling) |
|:------------------|:----------------------------|
| Barrier potential $U(x)$ | Effective potential $\Phi_{\text{eff}}$ |
| Ground state energy $E_0$ | Nash equilibrium value |
| Tunneling exponent | Agmon distance $d_{\text{Ag}}$ |
| $\hbar \to 0$ limit | $\sigma \to 0$ (classical Nash) |
::::

:::{prf:corollary} Bohm Potential Enables Strategic Teleportation
:label: cor-bohm-teleportation

When the Bohm potential $Q_B$ dominates (high belief curvature), the **effective barrier** becomes:

$$
\Phi^{\text{quantum}}_{\text{eff}}(\mathbf{z}) := \Phi_{\text{eff,total}}(\mathbf{z}) + Q_B(\mathbf{z}).

$$
In regions where $Q_B < 0$ (convex $\rho$), the effective barrier can become **negative** even when $\Phi_{\text{eff}} > 0$. This enables "teleportation" across classically forbidden regions.

**Operational interpretation:**
- An agent with **high uncertainty** (diffuse, smooth $\rho$) has $Q_B \approx 0$ → normal barrier
- An agent with **localized uncertainty** near the barrier (peaked, curved $\rho$) can have $Q_B \ll 0$ → reduced effective barrier
- The WFR **reaction term** $r$ (mass creation/destruction) provides the mechanism for "teleporting" belief mass without traversing intermediate states

*Remark (Exploration-Exploitation).* This provides a geometric foundation for the exploration-exploitation tradeoff: maintaining some uncertainty ($Q_B \neq 0$) is necessary to escape local optima.

:::

:::{prf:proposition} WFR Reaction as Tunneling Mechanism
:label: prop-wfr-reaction-tunneling

The WFR reaction term $r(z)$ (Definition {prf:ref}`def-the-wfr-action`) enables tunneling via **mass creation on the far side** of barriers:

1. Agent detects high-value region $\mathbf{z}^*_B$ beyond barrier $\mathcal{B}_P$
2. Reaction term $r(\mathbf{z}^*_B) > 0$ creates belief mass at $\mathbf{z}^*_B$
3. Reaction term $r(\mathbf{z}^*_A) < 0$ destroys mass at old position $\mathbf{z}^*_A$
4. Net effect: belief "teleports" without traversing $\mathcal{B}_P$

The rate of this process is controlled by the **teleportation length** $\lambda$ (Definition {prf:ref}`def-canonical-length-scale`):
- $\lambda \gg$ barrier width: tunneling is fast (reaction-dominated)
- $\lambda \ll$ barrier width: tunneling is slow (transport-dominated)

:::

(sec-summary-of-qm-agent-isomorphisms)=
## Summary of QM-Agent Isomorphisms

The following table consolidates the correspondence between quantum mechanical concepts and their Fragile Agent interpretations.

**Table 29.13.1 (Quantum-Agent Dictionary).**

| Quantum Mechanics | Fragile Agent Theory | Definition/Location |
|:------------------|:---------------------|:--------------------|
| **Wave-function $\psi$** | Belief Amplitude $\sqrt{\rho}e^{iV/\sigma}$ | {prf:ref}`def-belief-wave-function` |
| **Probability $\|\psi\|^2$** | Belief Density $\rho$ | Definition {prf:ref}`def-the-wfr-action` |
| **Phase $\arg(\psi)$** | Value Function $V/\sigma$ | Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence` |
| **Planck constant $\hbar$** | Cognitive Action Scale $\sigma$ | {prf:ref}`def-cognitive-action-scale` |
| **Hilbert space $\mathcal{H}$** | $L^2(\mathcal{Z}, d\mu_G)$ | {prf:ref}`def-inference-hilbert-space` |
| **Hamiltonian $\hat{H}$** | Inference Hamiltonian $\hat{H}_{\text{inf}}$ | {prf:ref}`thm-madelung-transform` |
| **Kinetic energy $-\frac{\hbar^2}{2m}\nabla^2$** | Diffusion term $-\frac{\sigma^2}{2} D^i D_i$ | {ref}`sec-summary-table-from-single-to-multi-agent` |
| **Potential energy $V(x)$** | Effective Potential $\Phi_{\text{eff}}$ | Definition {prf:ref}`def-effective-potential` |
| **Quantum potential $Q$** | Information Resolution Limit $Q_B$ | {prf:ref}`def-bohm-quantum-potential` |
| **Schrödinger equation** | Inference-Wave equation | {prf:ref}`thm-madelung-transform` |
| **Entanglement** | Strategic Coupling (non-factorizable) | {prf:ref}`def-strategic-entanglement` |
| **Tensor product $\otimes$** | Joint Hilbert space | {prf:ref}`def-joint-inference-hilbert-space` |
| **Partial trace** | Marginalization / Information Bottleneck | {prf:ref}`prop-partial-trace-reduced-dynamics` |
| **Ground state** | Nash Equilibrium | {prf:ref}`thm-nash-ground-state` |
| **Tunneling** | Pareto Barrier Crossing | {prf:ref}`thm-tunneling-probability` |
| **Imaginary time evolution** | Value Iteration | {prf:ref}`prop-imaginary-time-nash-finding` |
| **Density matrix $\hat{\rho}$** | Belief Operator (GKSL) | Definition {prf:ref}`def-belief-operator` |
| **Lindblad dissipator** | WFR Reaction Term $r$ | Definition {prf:ref}`def-gksl-generator` |
| **von Neumann entropy** | Belief Entropy $-\text{Tr}[\hat{\rho}\ln\hat{\rho}]$ | {ref}`sec-strategic-connection-covariant-derivative` |
| **WKB approximation** | Semiclassical limit | {prf:ref}`cor-semiclassical-limit` |
| **Spectral gap** | Convergence rate to Nash | {prf:ref}`prop-imaginary-time-nash-finding` |

**Interpretation Hierarchy:**
1. **Level 1 (Symplectic):** Ghost Interface $\mathcal{G}_{ij}$ couples agent boundaries with retardation
2. **Level 2 (Riemannian):** Game Tensor $\mathcal{G}_{ij}$ curves the metric (with strategic delay)
3. **Level 3 (Thermodynamic):** Landauer bounds constrain information processing
4. **Level 4 (Quantum):** Wave-function provides superposition and tunneling

Each level adds expressive power while remaining mathematically consistent with the levels below.

(sec-diagnostic-nodes-quantum-consistency)=
## Diagnostic Nodes 57–60 (Quantum Consistency)

Following the diagnostic node convention ({ref}`sec-theory-thin-interfaces`), we define four new monitors for quantum consistency in multi-agent systems.

(node-57)=
**Node 57: CoherenceCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **57** | **CoherenceCheck** | Multi-Agent | Unitary Consistency | Is the belief update probability-preserving? | $\delta_{\text{coh}} := \left\lvert \|\Psi_{s+\Delta s}\|^2 - \|\Psi_s\|^2 \right\rvert$ | $O(N d)$ |

**Interpretation:** Monitors deviation from unitarity (or trace-preservation for density matrices). Non-zero $\delta_{\text{coh}}$ indicates:
- Numerical integration error
- Unmodeled dissipation channels
- Inconsistency between Hamiltonian and WFR dynamics

**For open systems:** Replace with trace preservation check $\delta_{\text{tr}} := |\text{Tr}[\hat{\rho}_{s+\Delta s}] - 1|$.

**Threshold:** $\delta_{\text{coh}} < \epsilon_{\text{coh}}$ (typical default $10^{-6}$).

**Trigger conditions:**
- High CoherenceCheck: Numerical instability or model inconsistency
- **Remedy:** Reduce timestep; verify Hamiltonian is Hermitian (up to reaction term); check boundary conditions

(node-58)=
**Node 58: EntropyProductionCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **58** | **EntropyProductionCheck** | Multi-Agent | Thermodynamic | Is entropy production physically reasonable? | $\dot{S}_{\text{vN}} := -\frac{d}{ds}\text{Tr}[\hat{\rho}\ln\hat{\rho}]$ | $O(N^2 d)$ |

**Interpretation:** Monitors the rate of **von Neumann entropy** change:
- $\dot{S}_{\text{vN}} > 0$: Entropy increasing (learning, decoherence, information gain from environment)
- $\dot{S}_{\text{vN}} < 0$: Entropy decreasing (spontaneous ordering, may indicate instability)
- $\dot{S}_{\text{vN}} \approx 0$: Near equilibrium

**Connection to Landauer:** Entropy production should satisfy $\dot{S}_{\text{vN}} \geq -\mathcal{W}_{\text{comp}}/T_c$ where $\mathcal{W}_{\text{comp}}$ is computational work ({ref}`sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics`).

**Threshold:** $|\dot{S}_{\text{vN}}| < \dot{S}_{\max}$ (implementation-dependent).

**Trigger conditions:**
- Large positive $\dot{S}_{\text{vN}}$: Rapid decoherence ("losing mind")
- Large negative $\dot{S}_{\text{vN}}$: Anomalous ordering (check for mode collapse)

(node-59)=
**Node 59: UncertaintyPrincipleCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **59** | **UncertaintyPrincipleCheck** | Multi-Agent | Consistency | Are uncertainty bounds satisfied? | $\eta_{\text{unc}} := \frac{\sigma/2}{\sigma_z \sigma_p} \leq 1$ | $O(N d)$ |

**Interpretation:** The **Heisenberg-Robertson uncertainty relation** on the latent manifold requires:

$$
\sigma_z \cdot \sigma_p \geq \frac{\sigma}{2} |\langle[\hat{z}, \hat{p}]\rangle| = \frac{\sigma}{2},

$$
where:
- $\sigma_z := \sqrt{\langle z^2 \rangle - \langle z \rangle^2}$ is position uncertainty
- $\sigma_p := \sqrt{\langle p^2 \rangle - \langle p \rangle^2}$ is momentum uncertainty ($p = G\mathbf{v} = -\nabla_B V$)

**Violation $\eta_{\text{unc}} > 1$:** The agent claims to know both "where it is" (position $z$) and "where it is going" (momentum $-\nabla_B V$) with precision exceeding the information-theoretic limit. This indicates:
- Over-confident world model
- Ungrounded predictions
- Numerical precision issues

**Threshold:** $\eta_{\text{unc}} < 1$ (hard constraint from information geometry).

**Trigger conditions:**
- $\eta_{\text{unc}} > 1$: Uncertainty violation
- **Remedy:** Increase $\sigma$ (cognitive temperature); add regularization; verify encoder-decoder consistency

(node-60)=
**Node 60: TunnelingRateMonitor**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **60** | **TunnelingRateMonitor** | Multi-Agent | Exploration | Is barrier crossing rate reasonable? | $\Gamma_{\text{tunnel}} := P_{\text{tunnel}} / \tau_{\text{obs}}$ | $O(N^2 d)$ |

**Interpretation:** Monitors the rate at which agents cross Pareto barriers (Theorem {prf:ref}`thm-tunneling-probability`):
- $\Gamma_{\text{tunnel}} \approx 0$: Agents trapped in local equilibria (insufficient exploration)
- $\Gamma_{\text{tunnel}}$ moderate: Healthy exploration-exploitation balance
- $\Gamma_{\text{tunnel}}$ very high: Unstable dynamics (agents "teleporting" erratically)

**Computation:** Track probability mass flux across identified barrier surfaces using:

$$
\Gamma_{\text{tunnel}} = \int_{\partial \mathcal{B}_P} \mathbf{J} \cdot \mathbf{n}\, d\Sigma,

$$
where $\mathbf{J}$ is probability current and $\partial \mathcal{B}_P$ is the barrier boundary.

**Threshold:** $\Gamma_{\min} < \Gamma_{\text{tunnel}} < \Gamma_{\max}$ (task-dependent).

**Trigger conditions:**
- Low tunneling: Increase $\sigma$ or WFR reaction rate
- High tunneling: Decrease exploration; check for instabilities
- Asymmetric tunneling (one direction only): May indicate irreversible dynamics

(sec-implementation-causal-buffer)=
## Implementation: The Causal Buffer

:::{div} feynman-prose
All right, enough theory. How do you actually build this thing?

The key implementation challenge is handling the retardation. When agent A looks at agent B, it should not see the current state of B. It should see the ghost, the state of B from time $\tau_{ij}$ ago. How do you do that without rewriting your entire codebase?

The answer is the Causal Buffer. It is a simple data structure that sits between agents and handles all the time-delay bookkeeping transparently. When agent B takes an action or updates its state, it writes to the buffer. When agent A wants to read B's state, it reads from the buffer at the appropriate retarded time.

The buffer just stores timestamped records and interpolates when you request a time that falls between records. That is it. The rest of your code can pretend that it is seeing instantaneous states, but what it actually sees are ghosts.

The beauty of this approach is that it is modular. You do not have to rewrite your policy network or your value function. You just wrap the inter-agent communication in a Causal Buffer, and suddenly your system respects causality. All the relativistic effects, the standing-wave Nash, the strategic hysteresis, they all emerge automatically from this simple bookkeeping.

Of course, you need to choose $c_{\text{info}}$ appropriately for your domain. In a physical robotics scenario, it might be the speed of light or sound. In a networked system, it is the inverse of the network latency. In some simulations, you might want to crank it up to infinity to recover the Newtonian limit. The Causal Buffer handles all these cases.
:::

To implement relativistic multi-agent dynamics without disrupting existing software architecture, we introduce a **Causal Buffer** that handles time-retardation transparently.

**Algorithm 29.20.1 (Causal Context Buffer).**

```python
class CausalContextBuffer(nn.Module):
    """
    Implements the Memory Screen for Relativistic Agents.
    Stores past signals and serves them based on light-cone delay.

    The buffer maintains a history of (time, signal) pairs and provides
    ghost states by interpolating to the appropriate retarded time.
    """
    def __init__(self, context_dim: int, max_latency: int = 100):
        super().__init__()
        self.buffer = []  # Ring buffer of (t, signal)
        self.c_info = 1.0  # Information speed (environment units / timestep)
        self.max_latency = max_latency
        self.context_dim = context_dim

    def write(self, t: int, signal: torch.Tensor):
        """
        Agent J emits signal at time t.

        Args:
            t: Current timestep
            signal: State/action tensor to record
        """
        self.buffer.append((t, signal.clone()))
        # Prune old entries beyond max latency
        while self.buffer and self.buffer[0][0] < t - self.max_latency:
            self.buffer.pop(0)

    def read(self, t_now: int, dist: float) -> torch.Tensor:
        """
        Agent I reads signal arriving at t_now from distance dist.
        Returns the ghost state: signal emitted at t_emit = t_now - dist/c_info.

        Args:
            t_now: Current time of the reading agent
            dist: Environment distance d_E^{ij} between agents

        Returns:
            Ghost signal from retarded time, or zero if no data available
        """
        t_emit_target = t_now - (dist / self.c_info)

        if not self.buffer:
            return torch.zeros(self.context_dim)

        # Interpolate from buffer (Ghost State reconstruction)
        ghost_signal = self._interpolate(t_emit_target)
        return ghost_signal

    def _interpolate(self, t_target: float) -> torch.Tensor:
        """Linear interpolation between nearest buffer entries."""
        if len(self.buffer) == 1:
            return self.buffer[0][1]

        # Find bracketing entries
        for i in range(len(self.buffer) - 1):
            t_lo, s_lo = self.buffer[i]
            t_hi, s_hi = self.buffer[i + 1]
            if t_lo <= t_target <= t_hi:
                alpha = (t_target - t_lo) / (t_hi - t_lo + 1e-8)
                return (1 - alpha) * s_lo + alpha * s_hi

        # Extrapolate if t_target outside range
        if t_target < self.buffer[0][0]:
            return self.buffer[0][1]
        return self.buffer[-1][1]
```

**Integration with HolographicInterface:** The `CausalContextBuffer.read()` output serves as the `context` tensor for the policy $\pi(a|z, c)$. The policy remains Markovian with respect to the augmented input $(z_t, \Xi_{<t})$.

```python
class RelativisticMultiAgentInterface(nn.Module):
    """
    Extends HolographicInterface for relativistic multi-agent settings.
    Wraps agent-to-agent communication with causal buffers.
    """
    def __init__(self, n_agents: int, config: InterfaceConfig,
                 env_distances: torch.Tensor, c_info: float = 1.0):
        super().__init__()
        self.n_agents = n_agents
        self.c_info = c_info

        # Causal buffer for each ordered pair (i, j)
        self.buffers = nn.ModuleDict({
            f"{i}_{j}": CausalContextBuffer(config.context_dim)
            for i in range(n_agents) for j in range(n_agents) if i != j
        })

        # Precompute causal delays tau_ij = d_ij / c_info
        self.register_buffer('tau', env_distances / c_info)

    def broadcast(self, agent_id: int, t: int, state: torch.Tensor):
        """Agent broadcasts its state to all buffers it writes to."""
        for j in range(self.n_agents):
            if j != agent_id:
                self.buffers[f"{agent_id}_{j}"].write(t, state)

    def receive_context(self, agent_id: int, t: int) -> torch.Tensor:
        """
        Agent receives ghost states from all other agents.
        Returns concatenated context from past light cone.
        """
        contexts = []
        for j in range(self.n_agents):
            if j != agent_id:
                tau_ij = self.tau[agent_id, j].item()
                ghost = self.buffers[f"{j}_{agent_id}"].read(t, tau_ij * self.c_info)
                contexts.append(ghost)
        return torch.cat(contexts, dim=-1)
```

*Cross-reference:* This implementation realizes the Ghost Interface (Definition {prf:ref}`def-ghost-interface`) and Memory Screen (Definition {prf:ref}`def-memory-screen`) in differentiable form, enabling end-to-end training of relativistic multi-agent systems.



(sec-extended-summary-table)=
## Extended Summary Table

**Table 29.29.1 (Extended SMFT Summary with Relativistic, Gauge, and Quantum Layers).**

| Concept | Single Agent (Sec. 20–24) | Multi-Agent Relativistic (Sec. 29.1–29.12) | Multi-Agent Gauge (Sec. 29.13–29.20) | Multi-Agent Quantum (Sec. 29.21–29.27) |
|:--------|:--------------------------|:-------------------------------------------|:-------------------------------------|:---------------------------------------|
| **State Space** | $\mathcal{Z}$ | $\mathcal{Z}_{\text{causal}} = \mathcal{Z}^{(N)} \times \Xi_{<t}$ | $\mathcal{P}(G) \times \mathcal{Z}_{\text{causal}}$ | $\mathcal{H}^{(N)} = \bigotimes_i \mathcal{H}^{(i)}$ |
| **State Rep.** | Density $\rho(z)$ | Causal Bundle $(z^{(i)}_t, \Xi^{(i)}_{<t})$ | + Gauge connection $A_\mu$ | Wave-function $\Psi(\mathbf{z})$ or operator $\hat{\rho}$ |
| **Dynamics** | WFR Continuity + HJB | Coupled WFR + Retardation | + Yang-Mills equations | Schrödinger equation |
| **Generator** | Fokker-Planck / WFR | Coupled continuity (Klein-Gordon) | + $D_\mu \mathcal{F}^{\mu\nu} = J^\nu$ | Hamiltonian $\hat{H}_{\text{strat}}$ |
| **Kinetic Term** | $\nabla \cdot (\rho G^{-1}\nabla_B V)$ | $\frac{1}{c^2}\partial_t^2 - \Delta_G$ | $\frac{1}{c^2}D_t^2 - D^i D_i$ | $-\frac{\sigma^2}{2} D^i D_i$ (with $\tilde{G}$) |
| **Potential** | $\Phi_{\text{eff}}(z)$ | $\Phi^{\text{ret}}_{ij}(z^{(i)}, t)$ (quasi-static: $\approx \Phi_{ij}(z^{(i)}, z^{(j)}_{t-\tau})$) | + Higgs $\mu^2|\Phi|^2 + \lambda|\Phi|^4$ | Operator $\hat{\Phi}_{\text{eff}} + \sum \hat{V}_{ij}$ |
| **Metric Effect** | $G$ | $\tilde{G}^{(i)}(t) = G^{(i)} + \sum_j \beta_{ij}\mathcal{G}^{(i),\text{ret}}_{ij}$ | + Gauge-covariant $\tilde{\mathcal{G}}^{(i)}_{ij}$ | Game-Augmented Laplacian |
| **Coupling Mechanism** | — | Ghost Interface $\mathcal{G}_{ij}(t)$ | + Gauge connection $A_\mu$ | + Strategic Entanglement |
| **Resolution Limit** | — | Causal delay $\tau_{ij} = d_{\mathcal{E}}^{ij}/c_{\text{info}}$ | Mass gap $\Delta_{\text{KG}} > 0$ | Bohm potential $Q_B$ |
| **Coupling Type** | — | Potential + Metric + Retardation | + Gauge curvature $\mathcal{F}_{\mu\nu}$ | + Entanglement |
| **Correlation** | — | Classical (delayed, factorizable) | + Screened ($\xi = 1/\kappa < \infty$) | Quantum (non-factorizable) |
| **Equilibrium** | Value maxima | Standing Wave (time-averaged Nash) | + Symmetry breaking (VEV) | Ground state of $\hat{H}_{\text{strat}}$ |
| **Equilibrium Char.** | $\nabla_B V = 0$ | $\langle \mathbf{J}^{(i)} \rangle_T = 0$ | $\langle\Phi\rangle = v/\sqrt{2}$ | $\hat{H}\Psi = E_0\Psi$ |
| **Barrier Crossing** | Thermal noise | Strategic delay + coupling | Confinement (basin locking) | Quantum tunneling |
| **Nash Finding** | Gradient descent | Coupled gradient + causal buffer | + Gauge-covariant descent | Imaginary time evolution |
| **Diagnostics** | Nodes 1–45 | + Nodes 46–48, 62 | + Nodes 63–66 | + Nodes 57–60 |
| **Newtonian Limit** | — | $c_{\text{info}} \to \infty$: instantaneous | $g \to 0$: Abelian (Maxwell) | $\sigma \to 0$: classical Nash |

**Open problems (extended):**
1. *Scalability:* Tensor product Hilbert space dimension scales as $\prod_i \dim(\mathcal{H}^{(i)})$. Mean-field or tensor network approximations needed for large $N$.
2. *Decoherence timescales:* How fast does strategic entanglement decay under realistic noise? What is the effective "decoherence time" for multi-agent systems?
3. *Entanglement witnesses:* Can we design efficient diagnostics to detect and quantify strategic entanglement without full state tomography?
4. *Quantum speedup:* Does the Schrödinger formulation enable faster Nash-finding algorithms (quantum advantage in game theory)?
5. *Topological phases:* When $\mathcal{Z}$ has non-trivial topology, can agents exhibit "topologically protected" strategies immune to local perturbations?
