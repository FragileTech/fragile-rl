# Introduction: The Agent as a Bounded-Rationality Controller

## TLDR

- Treat the environment as an **input-output law** at an observation/action interface (a Markov blanket), not as a hidden
  state you can “recover”.
- Define the agent as a **bounded-rationality controller** with an internal state split into macro / nuisance / texture:
  $Z_t = (K_t, z_{n,t}, z_{\mathrm{tex},t})$.
- Re-type standard RL objects (state, observation, action, reward) as **boundary signals and constraints**, so stability
  and capacity can be enforced explicitly later (Sieve diagnostics/barriers).
- Make **symmetries and gauge freedoms** first-class: equivalence classes of representations matter more than raw
  coordinates.
- Separate notions of time (interaction, computation, scale, memory) to avoid category errors when connecting theory to
  implementation.

## Roadmap

1. Define interaction under partial observability and the boundary/Markov blanket view.
2. Introduce the internal state decomposition and why it is the right type signature for control.
3. Record symmetries, units, and time conventions that later chapters rely on.

## Symbols (Quick)

| Symbol | Meaning |
|---|---|
| $x_t$ | Observation at time $t$ (incoming boundary signal) |
| $a_t$ | Action at time $t$ (outgoing boundary signal) |
| $r_t$ | Reward sample (boundary flux; scalar in conservative case) |
| $d_t$ | Termination / absorbing event indicator (terminal subset $\Gamma_{\text{term}}$) |
| $\iota_t$ | Auxiliary side channels (costs, constraint reasons, privileged signals) |
| $B_t=(x_t,r_t,d_t,\iota_t,a_t)$ | Boundary / Markov blanket interface tuple |
| $Z_t=(K_t,z_{n,t},z_{\mathrm{tex},t})$ | Internal state split (macro / nuisance / texture) |
| $P_{\partial}$ | Environment as an input-output law over boundary signals |

:::{div} feynman-prose
Let me start with a confession. When most people think about artificial intelligence or reinforcement learning, they imagine a perfect reasoner---something that can consider all possibilities, hold infinite information in its head, and compute instantly. That's a beautiful mathematical fiction, and it has its uses, but it's not what we're going to talk about here.

We're going to talk about *real* agents. Agents that can only see some of what's happening. Agents that can only remember so much. Agents that have to make decisions *now*, before they've finished thinking. In other words: agents that are *bounded*. This notion of bounded rationality---that real decision-makers must satisfice rather than optimize due to computational and informational constraints---was formalized by Herbert Simon {cite}`simon1955behavioral,simon1982models`.

This isn't a limitation we're going to apologize for and then sweep under the rug. This is the central fact we're going to build our entire framework around. Because here's the thing---every interesting agent, biological or artificial, is bounded. Your brain doesn't have infinite capacity. Neither does GPT-4. Neither does any robot we'll ever build. So we'd better have a theory that takes this seriously.
:::

(rb-bounded-rationality)=
:::{admonition} Researcher Bridge: Bounded Rationality as a POMDP with Costs
:class: info
Standard RL frames the agent as a policy that maximizes return in a POMDP. Here we make the usual hidden constraints explicit: limited bandwidth, memory, and compute. Think of it as a POMDP with an information bottleneck and hard safety contracts that shape the feasible policy class.
:::

:::{div} feynman-prose
This document presents the **Fragile** interpretation of the Hypostructure: a deployed agent is a persistent **controller under partial observability** whose competence is bounded by (i) finite sensing/communication bandwidth, (ii) finite internal memory/representation capacity, and (iii) finite compute for inference and planning.

The framework is stated strictly in **information theory, optimization, and control**: discrete/continuous latent state construction (representation), stability constraints (Lyapunov-style), and capacity/sufficiency conditions (information bottlenecks).

This is the native language of **Safe RL**, **Robust Control**, and **Embodied AI**.
:::

(sec-definitions-interaction-under-partial-observability)=
## Definitions: Interaction Under Partial Observability

:::{div} feynman-prose
Now, here's where things get philosophically interesting---and I mean that in a practical way, not a hand-wavy way.

When you learn reinforcement learning from a textbook, you typically start with something like: "There is an environment. The environment has states. The agent observes these states and takes actions." And that's fine as far as it goes. But there's a hidden assumption in there that causes all sorts of trouble: the idea that we know what the environment "really is."

We don't. We can't. The only access an agent has to the world is through its sensors and actuators. Everything else is inference.
:::

(rb-markov-blanket)=
:::{admonition} Researcher Bridge: Markov Blanket = Observation/Action Interface
:class: tip
If you are used to POMDP notation, the "boundary" here is just the observation, action, reward, and termination channels treated as a single interface. The environment is an input-output law, not a latent object the agent can access directly. This re-typing lets us attach geometric and information constraints to the interface itself.
:::

:::{div} feynman-prose
In the Fragile Agent framework, we do **not** treat the environment as a passive data provider. We treat the agent as a **partially observed control problem** whose only coupling to the external world is through a well-defined **interface / Markov blanket**. All RL primitives are re-typed as **signals and constraints at this interface**.
:::

(sec-the-environment-is-an-input-output-law)=
### The Environment is an Input-Output Law (Not an Internal Object)

:::{div} feynman-prose
Let me make this very concrete. Imagine you're a robot arm in a factory. You have cameras, force sensors, joint encoders. You can move your motors. That's it. That's all you've got.

Now, somewhere out there is a "world"---conveyor belts moving, objects on the belts, other robots, temperature fluctuations, everything. But you don't see any of that directly. You see pixels from cameras. You feel forces on sensors. You hear the whine of your own motors.

The key insight is this: *from the agent's perspective, the environment is not a thing with internal states that it can inspect.* The environment is a black box that responds to your actions with new observations. It's a function, or really a stochastic process: you put actions in, you get observations out.

This might seem like a mere philosophical nicety, but it has profound practical implications. If you think of the environment as having "true states" that you're trying to learn, you'll build systems that try to recover those states. But you *can't* recover them---you can only build internal representations that are useful for prediction and control. That's a different problem, and it leads to different (better) algorithms.
:::

:::{prf:definition} Bounded-Rationality Controller
:label: def-bounded-rationality-controller

The agent is a controller with internal state

$$
Z_t := (K_t, z_{n,t}, z_{\mathrm{tex},t}) \in \mathcal{Z}=\mathcal{K}\times\mathcal{Z}_n\times\mathcal{Z}_{\mathrm{tex}},

$$
and internal components (Encoder/Shutter, World Model, Critic, Policy). Its evolution is driven only by the observable interaction stream at the interface (observations/feedback) and by its own outgoing control signals (actions).

:::

:::{div} feynman-prose
Notice what this definition *doesn't* say. It doesn't say the agent has access to the "true state of the world." It says the agent has an *internal state* $Z_t$---its own representation, built from what it has observed. And that internal state has structure: a discrete macro-state $K_t$ (think: "what kind of situation is this?"), a nuisance component $z_{n,t}$ (think: "where exactly am I within this situation?"), and a texture component $z_{\text{tex},t}$ (think: "fine details needed for reconstruction but not for decision-making").

We'll unpack this decomposition thoroughly later. For now, the key point is that the agent's state is *its own construction*, not a window into some external truth.
:::

:::{prf:definition} Boundary / Markov Blanket
:label: def-boundary-markov-blanket

The boundary variables at time $t$ are the interface tuple

$$
B_t := (x_t,\ r_t,\ d_t,\ \iota_t,\ a_t),

$$
where:
- $x_t\in\mathcal{X}$ is the observation (input sample),
- $r_t\in\mathbb{R}$ is the boundary reward sample (evaluation of the reward 1-form/flux; scalar in the conservative case),
- $d_t\in\{0,1\}$ is termination (absorbing event / task boundary; corresponds to $\Gamma_{\text{term}}$ and $\tau_{\text{term}}$ in Definition {prf:ref}`def-terminal-boundary`),
- $\iota_t$ denotes any additional side channels (costs, constraints, termination reasons, privileged signals),
- $a_t\in\mathcal{A}$ is action (control signal sent outward).

:::

:::{div} feynman-prose
This is the agent's "skin." Everything the agent knows about the outside world comes through this interface. Everything the agent does to the outside world goes through this interface. The tuple $B_t$ is the complete accounting of the agent's interaction with reality at time $t$.

Why call it a "Markov blanket"? The term comes from graphical models and the theory of self-organizing systems {cite}`kirchhoff2018markov`. The Markov blanket of a node in a probabilistic graph is the minimal set of other nodes that, if you knew their values, would tell you everything you could possibly learn about that node from the rest of the graph. Conditioned on the blanket, the node is independent of everything outside.

For our agent, the boundary $B_t$ plays exactly this role. If you tell me the complete history of boundary variables $(B_1, B_2, \ldots, B_t)$, I know everything the agent could possibly know. There's no other channel of information. The "environment" is whatever is on the other side of this blanket---and from the agent's perspective, all that matters about the environment is how it responds to actions with new observations.
:::

:::{prf:definition} Environment as Generative Process
:label: def-environment-as-generative-process

The "environment" is the conditional law of future interface signals given past interface history. Concretely it is a (possibly history-dependent) kernel on incoming boundary signals conditional on the boundary history:

$$
P_{\partial}(x_{t+1}, r_{t+1}, d_{t+1}, \iota_{t+1}\mid B_{\le t}).

$$
In the Markov case this reduces to the familiar RL kernel

$$
P_{\partial}(x_{t+1}, r_{t+1}, d_{t+1}, \iota_{t+1}\mid B_t),

$$
but the **interpretation changes**: $P_{\partial}$ is not "a dataset generator"; it is the **input-output law** that the controller must cope with under partial observability and model mismatch.

This is the categorical move: we do not assume access to the environment's latent variables; we work only with the **law over observable interface variables**.

:::

:::{div} feynman-prose
Let me say that again because it's important: the environment is not a thing, it's a *law*. A conditional probability distribution. A stochastic function. You put in actions, you get out observations, and that's all you can ever know about it.
:::

:::{admonition} Why This Matters
:class: note

You might think: "Fine, philosophically the agent only sees observations. But surely in practice we can just use the 'true state' from the simulator?"

Here's why that's dangerous:

1. **Sim-to-real transfer**: Your simulator has latent states, but reality doesn't give you access to them. If your agent is trained to rely on true states, it won't transfer.

2. **Model mismatch**: Even if you had access to "true states," your model of the dynamics is approximate. Treating observations as the fundamental object forces you to be honest about this.

3. **Adversarial robustness**: The environment might be adversarial. If you assume you know its internal states, an adversary can exploit that assumption.

4. **Information constraints**: Real agents have limited capacity to process information. Working at the interface level lets you explicitly track information flow.

The input-output law perspective isn't a philosophical indulgence---it's an engineering requirement for building robust systems.
:::

:::{admonition} Worked Example: A POMDP as Boundary Signals + Split Latents
:class: tip

Start from the standard POMDP objects $(s_t, o_t, a_t, r_t)$ (latent state, observation, action, reward).

- **Boundary signals:** identify $x_t := o_t$ and form the boundary tuple $B_t := (x_t, r_t, d_t, \iota_t, a_t)$, where
  $d_t$ is termination and $\iota_t$ collects any extra side channels (costs, constraint reasons, privileged info).
- **Environment as law:** instead of “learning $s_t$”, treat the environment as the conditional law
  $P_{\partial}(x_{t+1}, r_{t+1}, d_{t+1}, \iota_{t+1}\mid B_{\le t})$.
- **Internal split:** choose/learn $Z_t=(K_t,z_{n,t},z_{\mathrm{tex},t})$ so that $K_t$ carries the *predictive,
  control-relevant* information, $z_{n,t}$ captures structured continuous variation (pose/disturbance), and
  $z_{\mathrm{tex},t}$ carries reconstruction-only detail.

Concrete intuition (robot navigation):
- $K_t$: “which room / which mode”.
- $z_{n,t}$: pose $(x,y,\theta)$ and other controllable continuous coordinates.
- $z_{\mathrm{tex},t}$: pixel-level texture that helps reconstruct $o_t$ but should not be required for control.
:::

(sec-re-typing-standard-rl-primitives-as-interface-signals)=
### Re-typing Standard RL Primitives as Interface Signals

:::{div} feynman-prose
Now let's go through the standard RL vocabulary and see how each term looks from this interface perspective. This isn't just relabeling---it's a change in how we think about these objects, and that change has consequences.
:::

1. **Environment (Stochastic Process / Unmodeled Disturbance).**
   - *Standard:* a black box providing states and rewards.
   - *Fragile:* a high-dimensional, partially observed stochastic process; only its induced interface law $P_{\partial}$ is accessible to the agent.
   - *Role:* supplies observations and feedback signals; may be non-stationary, adversarial, or only approximately Markov.

:::{div} feynman-prose
   Think about what this means. The "environment" isn't some ground truth we're trying to approximate. It's a source of uncertainty we're trying to cope with. It might be playing against us. It might be changing its rules. All we can do is learn patterns in how it responds to our actions.
:::

2. **Observation $x_t$ (Observation Stream).**
   - *Standard:* an input tensor.
   - *Fragile:* the only exogenous input available to the controller. The encoder/shutter transduces it into internal coordinates:

     $$
     x_t \mapsto (K_t, z_{n,t}, z_{\mathrm{tex},t}),

     $$
     where $K_t$ is the **discrete predictive signal** (bounded-rate latent statistic), $z_{n,t}$ is a **structured nuisance / gauge residual** (pose/basis/disturbance coordinates), and $z_{\mathrm{tex},t}$ is a **texture residual** (high-rate reconstruction detail).
   - *Boundary gate nodes ({ref}`sec-diagnostics-stability-checks`):*
     - **Node 14 (InputSaturationCheck):** input saturation (sensor dynamic range exceeded).
     - **Node 15 (SNRCheck):** low signal-to-noise (SNR too low to support stable inference).
     - **Node 13 (BoundaryCheck):** the channel is open in the only well-typed sense: $I(X;K)>0$ (symbolic mutual information).

:::{div} feynman-prose
   The observation isn't just "data." It's the only window the agent has into the world. And notice the decomposition: we immediately split what we see into "what kind of thing is this" ($K$), "where/how is it positioned" ($Z_n$), and "fine details for reconstruction" ($Z_{\text{tex}}$). This structured representation is crucial for efficiency and robustness.
:::

3. **Action $a_t$ (Control / Actuation).**
   - *Standard:* a vector sent to the environment.
   - *Fragile:* a control signal chosen to minimize expected future cost under uncertainty and constraints. Like observations, actions decompose into structured components: $a_t = (K^{\text{act}}_t, z_{n,\text{motor}}, z_{\text{tex,motor}})$ where $K^{\text{act}}_t$ is the discrete motor macro, $z_{n,\text{motor}}$ is motor nuisance (compliance), and $z_{\text{tex,motor}}$ is motor texture (tremor). See {ref}`sec-motor-texture-the-action-residual` for details.
   - *Cybernetic constraints:*
     - **Node 2 (ZenoCheck):** limits chattering (bounded variation in control outputs).
     - **BarrierSat:** actuator saturation (finite control authority).
   - *Boundary interpretation ({ref}`sec-the-symplectic-interface-position-momentum-duality`):* Actions impose **Neumann boundary conditions** (clamping flux/momentum) on the agent's internal manifold, dual to the Dirichlet conditions imposed by sensors.

:::{div} feynman-prose
   Actions aren't free. You can't jitter infinitely fast (Zeno check). You can't push infinitely hard (saturation). And there's a beautiful duality here: sensors tell you "where you are" (Dirichlet), while actions tell you "which way you're pushing" (Neumann). Together they close the boundary conditions on the agent's internal dynamics.
:::

4. **Reward $r_t$ (Utility / Negative Cost Signal).**
   - *Standard:* a scalar to maximize.
   - *Fragile:* boundary reward flux (a 1-form) evaluated along the trajectory. In continuous time it appears as an
     instantaneous **cost rate** $r_t=\langle\mathcal{R},\dot{z}\rangle$; in discrete time it appears as an
     incremental term in the Bellman/HJB consistency relation ({ref}`sec-the-hjb-correspondence`).
   - *Mechanism:* the critic's $V$ is the internal value/cost-to-go for the **exact (conservative) component**; the
     non-conservative component remains as a curl/connection term in the dynamics ({ref}`sec-the-reward-1-form`).
   - *Boundary interpretation ({ref}`sec-the-reward-1-form`):* Reward is a boundary reward flux
     1-form $J_r$ defined by the pullback $J_r=\iota^*\mathcal{R}$ of the bulk reward 1-form $\mathcal{R}$.
     By Hodge decomposition, $\mathcal{R} = d\Phi + \delta \Psi + \eta$; only the exact part $d\Phi$ (identified with
     $dV$, the critic's exact component) yields a scalar charge density $\sigma_r$ and a **Screened Poisson Equation**
     (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`). Define the non-exact component
     $A := \delta\Psi + \eta$, so $\mathcal{R} = d\Phi + A$ and $dA = \mathcal{F}$. The solenoidal/harmonic parts
     encode path-dependent reward and are carried by the connection/field strength.

:::{div} feynman-prose
   I want you to really think about this one. Reward isn't a magical signal from God telling you what's good. It's just
   another input---a boundary flux sampled as a scalar $r_t$. The agent has to figure out what to do with it. The
   conservative component propagates inward as a potential landscape, while the non-conservative component drives
   circulation and direction-dependent gains.
:::

5. **Termination $d_t$ (Absorbing Boundary Event).**
   - *Standard:* end-of-episode flag.
   - *Fragile:* an absorbing event: the trajectory has entered a terminal region (failure/success) or exited the modeled
     domain. Formally this is a terminal subset $\Gamma_{\text{term}}$ with stopping time $\tau_{\text{term}}$ and
     Dirichlet data $V|_{\Gamma_{\text{term}}}=V_{\text{term}}$ (Definition {prf:ref}`def-terminal-boundary`).

6. **Episode / Rollout (Finite-Horizon Segment).**
   - *Standard:* a finite trajectory segment.
   - *Fragile:* a finite window used to estimate a cumulative objective under uncertainty. "Success" is satisfying task constraints while maintaining stability; "failure" is crossing a monitored limit ({ref}`sec-4-limits-barriers-the-limits-of-control`).

(sec-symmetries-and-gauge-freedoms)=
### Symmetries and Gauge Freedoms (Operational)

:::{div} feynman-prose
Now we come to something that causes endless headaches in practice: the problem of *irrelevant variation*.

Imagine training a robot to pick up a mug. The mug can be in different positions, at different angles, under different lighting. These variations don't change the fundamental task---it's still "pick up the mug"---but a naive learning system will treat each variation as a different situation, requiring separate learning.

This is wildly inefficient. Worse, it makes the system brittle: show it a lighting condition it hasn't seen, and it falls apart.

The right way to think about this is through the lens of *symmetry*. If rotating the camera by 10 degrees doesn't change what action you should take, then rotation is a *symmetry* of the problem. If shifting the reward by a constant doesn't change the optimal policy, then reward shifts are a *symmetry*. These symmetries define equivalence classes: different observations that should map to the same decision.

Many of the largest stability and sample-efficiency failures in practice come from **wasting capacity on nuisance degrees of freedom**: the agent learns separate internal states for observations that differ only by pose, basis choice, or arbitrary internal labeling. We formalize these nuisance directions as **symmetries** (group actions) and treat "quotienting them out" as an explicit design constraint {cite}`cohen2016group`.

We use "gauge" in the minimal, operational sense: a **gauge transformation** is a change of coordinates or representation that should not change the agent's control-relevant decisions, except through explicitly modeled nuisance variables.

The word "gauge" comes from physics, where it refers to transformations that change the mathematical description without changing the physical predictions. That's exactly what we mean here: gauge transformations change how we *represent* the situation without changing what we should *do* about it.
:::

:::{prf:definition} Agent symmetry group; operational
:label: def-agent-symmetry-group-operational

Let:
- $G_{\text{obj}}$ be an **objective/feedback gauge** acting on scalar feedback signals (e.g., change of units or baseline shift). A common choice is the positive affine group

  $$
  G_{\text{obj}} := \{(a,b): a>0,\ r\mapsto ar+b\}.

  $$
  (If representing value as a unit-norm phase variable, one may instead use $U(1)$; {ref}`sec-defect-functionals-implementing-regulation`.C treats the real-valued case via projective heads.)
- $G_{\text{spatial}}$ be an **observation gauge** acting on raw observations $x$ (e.g., pose/translation/rotation; choose $SE(3)$, $SE(2)$, $\mathrm{Sim}(2)$, or a task-specific subgroup depending on sensors).
- $S_{|\mathcal{K}|}$ be the **symbol-permutation symmetry** of the discrete macro register: relabeling code indices is unobservable if downstream components depend only on embeddings $\{e_k\}$.
- $\mathrm{Symp}(2n,\mathbb{R})$ be an optional **phase-space symmetry** acting on canonical latent coordinates $z=(q,p)\in\mathbb{R}^{2n}$ when the world model is parameterized as a symplectic/Hamiltonian system {cite}`greydanus2019hamiltonian` ({ref}`sec-defect-functionals-implementing-regulation`.B).

The (candidate) total symmetry group is the direct product

$$
\mathcal{G}_{\mathbb{A}}
:=
G_{\text{obj}}
\times
G_{\text{spatial}}
\times
S_{|\mathcal{K}|}
\times
\mathrm{Symp}(2n,\mathbb{R}).

$$
**Internal vs. external symmetries.**
- **Internal (objective) gauge:** transformations of the scalar feedback scale/offset (and any potentials) that should not qualitatively change the policy update direction.
- **External (observation) gauge:** transformations of the input stream that change *pose* but not *identity*.

**Principle of covariance (engineering requirement).** The internal maps of the agent should be invariant/equivariant under $\mathcal{G}_{\mathbb{A}}$ in the following typed sense:
- **Shutter $E$**: canonicalize or quotient $G_{\text{spatial}}$ before discretization, so the macro register is approximately invariant:

  $$
  K(x)\approx K(g\cdot x)\quad (g\in G_{\text{spatial}}),

  $$
  while $z_n$ carries structured nuisance parameters (pose/basis/disturbance coordinates) and $z_{\mathrm{tex}}$ carries reconstruction-only texture ({ref}`sec-the-shutter-as-a-vq-vae`, {ref}`sec-defect-functionals-implementing-regulation`.A).
- **World model $S$ and policy $\pi$:** be covariant to symbol permutations $S_{|\mathcal{K}|}$ by treating $K$ only through its embedding $e_K$ (not the integer label) and by using permutation-invariant diagnostics.
- **Critic/value and dual variables:** enforce stability and constraint satisfaction in a way that is robust to re-scaling/offset of the scalar feedback ({ref}`sec-defect-functionals-implementing-regulation`.C, {ref}`sec-adaptive-multipliers-learned-penalties-setpoints-and-calibration`).

These are *requirements on representations and interfaces*, not philosophical claims: if an invariance is not enforced, the corresponding failure modes (symmetry blindness, brittle scaling, uncontrolled drift) become more likely and harder to debug.

:::

:::{admonition} The Point of Gauge Invariance
:class: tip

Let me give you a very concrete example of why this matters.

Suppose you train a vision-based robot with reward values in the range $[0, 100]$. Now you deploy it in a new setting where rewards are in $[-50, 50]$. If your critic isn't gauge-invariant to affine transformations of reward, it might completely misinterpret the situation.

Or suppose you train on images with the camera in one position, then the camera gets bumped. If your encoder isn't approximately equivariant to translations, the whole system might collapse.

Gauge invariance isn't fancy math for its own sake. It's the mathematical statement of: "The system should be robust to things that shouldn't matter." And by making the symmetries explicit, we can *enforce* them architecturally rather than hoping the network learns them.
:::

(sec-units-and-dimensional-conventions)=
### Units and Dimensional Conventions (Explicit)

:::{div} feynman-prose
Now, a brief but important aside about units. This might seem pedantic, but confusing units is one of the most common sources of bugs in scientific code. (NASA lost a Mars orbiter because of a unit confusion {cite}`nasa1999mars`. If it can happen to rocket scientists, it can happen to you.)
:::

This document expresses objectives in **information units** so that likelihoods, code lengths, KL terms, and entropy regularizers share a common scale.

**Base units.**
- Interaction time is measured in **environment steps** ($t \in \mathbb{Z}_{\ge 0}$). If a physical clock is needed, introduce $\Delta t$ with $[\Delta t]=\mathrm{s}$.
- Information: entropies/information/KL are in **nats** (dimensionless but tracked): $[H]=[S]=[I]=[D_{\mathrm{KL}}]=\mathrm{nat}$.
- Costs / values / losses (including $V$ and negative rewards) are measured in **nats**: $[V]=\mathrm{nat}$.
- Cost rates are measured in $\mathrm{nat/step}$ (or $\mathrm{nat\,s^{-1}}$ after dividing by $\Delta t$).

:::{div} feynman-prose
Why nats? Because natural logarithms are easier to work with in calculus ($d/dx \ln x = 1/x$), and because most of our information-theoretic quantities come from log-probabilities, which are naturally in nats. If you prefer bits, multiply by $\log_2 e \approx 1.44$.
:::

**Discrete vs continuous reward.**
- Per-step reward $r_t$ (or cost $c_t=-r_t$) has units $\mathrm{nat}$.
- A continuous-time cost rate $\mathcal{R}$ has units $\mathrm{nat\,s^{-1}}$ and links to discrete time by $r_t \approx \int_{t}^{t+\Delta t}\mathcal{R}(u)\,du$.

**Regularization / precision coefficients.**
- MaxEnt / entropy-regularized control introduces a trade-off coefficient (often written $T_c$ or $\alpha_{\text{ent}}$) multiplying an entropy term. Because entropy is in nats, this coefficient is dimensionless and simply sets relative weight in the objective.
- Exponential-family (softmax/logit) policies use a dimensionless precision parameter $\beta_{\text{ent}}$ so that $\exp(\beta_{\text{ent}}\,\cdot)$ is dimensionless; $\beta_{\text{ent}}$ is an inverse-variance / "sharpness" control knob.
- The metric coupling coefficient $\beta_{\text{cpl}}$ (Definition {prf:ref}`def-local-conditioning-scale`) carries units $\mathrm{nat}/[z]^2$, and the curl coefficient $\beta_{\text{curl}}$ appears in the dynamics.

**Conventions for generic coefficients.**
- Numerical stabilizers like $\epsilon$ always inherit the units of the quantity they are added to.
- Composite-loss weights (e.g. $\lambda_{\text{*}}$ used to sum training losses) are taken dimensionless unless explicitly stated otherwise.

(sec-the-chronology-temporal-distinctions)=
### The Chronology: Temporal Distinctions

:::{div} feynman-prose
Now we come to something that trips up almost everyone, and it's worth spending some time on it because getting this wrong leads to category errors that are very hard to debug.

The question is: what do we mean by "time"?

You might think this is obvious. Time is time, right? The clock ticks, things happen, one event follows another. But here's the thing: in a learning agent, there are *several different* notions of time, and they are completely distinct. Confusing them is like confusing spatial position with temperature---they're both numbers, but they measure entirely different things.

Let me give you an analogy. Imagine you're reading a book about history. There's the time *in the story* (1776, 1812, 1945). There's the time it takes you to read a page (maybe 2 minutes). There's the date when the book was written (2015). And there's the date when you're reading it (today). These are all "times," but they're completely different dimensions. You wouldn't ask "What happened on page 47?" and expect an answer like "July 4th, 1776." That's a category error.

In our agent, we have four different times, and they're analogous to these different notions:
:::

We distinguish four temporal dimensions. They are orthogonal (or nested) and must not be conflated. Using one symbol for all of them is a chronological category error (e.g., confusing "thinking longer" with "getting older").

| Symbol     | Name                 | Domain                           | Role                                              | Physics Analogy                  |
|:-----------|:---------------------|:---------------------------------|:--------------------------------------------------|:---------------------------------|
| **$t$**    | **Interaction Time** | $\mathbb{Z}_{\ge 0}$             | External environment clock ($x_t, a_t$).          | Coordinate time (observer clock) |
| **$s$**    | **Computation Time** | $\mathbb{R}_{\ge 0}$             | Internal solver time for belief/planning updates. | Proper time (agent thinking)     |
| **$\tau$** | **Scale Time**       | $\mathbb{R}_{\ge 0}$             | Resolution depth (root to leaf).                  | Renormalization scale            |
| **$t'$**   | **Memory Time**      | $\{t' \in \mathbb{Z} : t' < t\}$ | Index of stored past states on the screen.        | Retarded time                    |

:::{div} feynman-prose
Let me explain each one.
:::

(sec-interaction-time-the-discrete-clock)=
#### Interaction Time ($t$): The Discrete Clock

:::{div} feynman-prose
This is the most familiar one. It's the "game turn" or "time step" that you see in standard MDP/POMDP notation. At each tick $t$, the environment produces an observation, the agent produces an action, and we move on to $t+1$.
:::

This is the Markov Decision Process index imposed by the environment.
- **Update:** $z_t \to z_{t+1}$.
- **Constraint:** the agent must emit $a_t$ before $t$ increments (real-time constraint).

:::{div} feynman-prose
Think of this as the external clock---the metronome that the world imposes on the agent. The agent doesn't control how fast it ticks; it just has to keep up.
:::

(sec-computation-time-the-continuous-thought)=
#### Computation Time ($s$): The Continuous Thought

:::{div} feynman-prose
Now here's something subtle. Between receiving observation $x_t$ and emitting action $a_t$, the agent has to *think*. It has to update its beliefs, consider options, maybe do some planning. That thinking process takes time---not wall-clock time necessarily, but computational steps.

We parameterize this internal process with a continuous variable $s$. The agent starts at $s=0$ when it receives the observation, and by the time $s$ reaches some budget $S_{\text{budget}}$, it needs to have an action ready.
:::

This is the integration variable of the internal solver and the Equation of Motion ({ref}`sec-the-equations-of-motion-geodesic-jump-diffusion`). It represents the agent's "thinking" process:

$$
\frac{dz}{ds} = \mathcal{M}_{\text{curl}}\!\left(-G^{-1}\nabla \Phi_{\text{eff}} + \dots\right), \qquad \mathcal{M}_{\text{curl}} := (I - \beta_{\text{curl}} G^{-1}\mathcal{F})^{-1}

$$
- **Relationship to $t$:** to transition from $t$ to $t+1$, the agent integrates its internal dynamics from $s=0$ to $s=S_{\text{budget}}$.
- **Thinking fast vs. slow:** small $S_{\text{budget}}$ yields reflexive action; large $S_{\text{budget}}$ yields deliberate planning.
- **Thermodynamics:** this is the time variable in which Fokker-Planck dynamics evolve internal belief toward equilibrium ({ref}`sec-the-overdamped-limit`).

:::{div} feynman-prose
Here's the key insight: you can "think longer" within a single time step. An agent with more compute budget (larger $S_{\text{budget}}$) can do more deliberation before acting. This is like a chess player who takes 5 minutes to think versus one who moves immediately---they're both making one move (one increment of $t$), but one is doing more internal computation (larger $s$).
:::

:::{admonition} System 1 vs System 2
:class: note

This distinction between $t$ and $s$ gives us a natural way to think about fast vs. slow thinking. With $S_{\text{budget}} \approx 0$, the agent is purely reactive---it just maps observations to actions without deliberation. As $S_{\text{budget}}$ increases, the agent can do more planning, more model-based reasoning, more careful consideration of alternatives.

Daniel Kahneman's "Thinking, Fast and Slow" is essentially about this distinction. System 1 (fast, automatic, intuitive) corresponds to small $s$. System 2 (slow, deliberate, analytical) corresponds to large $s$. Our framework makes this quantitative.
:::

(sec-scale-time-the-holographic-depth)=
#### Scale Time ($\tau$): The Holographic Depth

:::{div} feynman-prose
This one is more exotic, and you might not have encountered it before. It's the notion of "resolution" or "level of detail" in a hierarchical representation.

Imagine you're looking at a photograph. You can look at the overall scene (coarse), or zoom in on a face (medium), or zoom in further on the texture of skin (fine). These different "zoom levels" aren't different times in the ordinary sense, but they form a kind of progression: from coarse to fine, from abstract to concrete.

In our framework, this becomes a formal dimension. We use $\tau$ to parameterize how "deep" we are in the representation hierarchy.
:::

This is the radial coordinate in the Poincare disk (Sections 21, 7.12). It corresponds to resolution depth.
- **Dynamics:** $dr/d\tau = \tfrac{1}{2}\operatorname{sech}^2(\tau/2)$ (the holographic law).
- **Discretization:** in stacked TopoEncoders, layer $\ell$ corresponds to scale time $\tau_\ell$.
- **Direction:** $\tau \to \infty$ (UV) is high energy, fine detail; $\tau \to 0$ (IR) is low energy, coarse structure.
- **Process:** generation flows in $+\tau$ (root to boundary); inference flows in $-\tau$.

:::{div} feynman-prose
Think of it this way: if you're generating an image, you start with a rough sketch ($\tau$ small) and progressively add detail ($\tau$ large). If you're analyzing an image, you start with the raw pixels ($\tau$ large) and progressively extract more abstract features ($\tau$ small). The variable $\tau$ indexes where you are in this hierarchy.

The "holographic" terminology comes from an analogy with physics, where information about a volume can be encoded on its boundary. Here, information about fine-scale structure is "encoded" in the boundary of our hierarchical representation.
:::

(sec-memory-time-the-historical-coordinate)=
#### Memory Time ($t'$): The Historical Coordinate

:::{div} feynman-prose
Finally, the agent has a memory. It remembers things that happened at previous time steps. We need a way to index into this memory.

That's what $t'$ is: an index into the past. If the current time is $t$, then $t'$ ranges over $\{0, 1, \ldots, t-1\}$---all the past moments that might be stored in memory.
:::

This is the time coordinate of the Holographic Screen.
- **Structure:** the screen stores tuples $(z_{t'}, a_{t'}, r_{t'})$ at past indices.
- **Access:** attention computes distances between the current state $z_t$ and stored states $z_{t'}$.
- **Causality:** we enforce $t' < t$ (no access to the future).

:::{div} feynman-prose
This is like asking "What was I thinking at step 47?" The answer depends on what was stored in memory at that time, and whether it's still accessible now.
:::

:::{admonition} The Category Error
:class: warning

Here's the concrete danger of confusing these times. Suppose someone says "Let's do more gradient steps to improve the agent."

Do they mean:
- More environment steps $t$? (Roll out longer trajectories)
- More computation steps $s$? (More planning iterations per action)
- More scale steps $\tau$? (Deeper hierarchical representations)
- More memory slots $t'$? (Longer context windows)

These are completely different interventions with completely different effects! Saying "we need more time" is meaningless until you specify which time dimension you're talking about.

In the equations that follow, you'll see these different time indices appearing in different places. Keeping them straight is essential for understanding what the equations actually say.
:::
