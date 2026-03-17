(sec-section-non-local-memory-as-self-interaction-functional)=
# Non-Local Memory as Self-Interaction Functional

## TLDR

- Memory is modeled as a **non-local force**: the past acts back on the present via a self-interaction functional, not a
  passive buffer.
- Introduce the **memory screen / historical manifold** needed to restore Markov structure when information propagates
  with delay.
- This chapter reframes experience replay and retrieval as geometric objects: kernels on history with stability
  diagnostics.
- Use the formalism to detect pathological memory behavior (stale recall, runaway attraction to a single trajectory,
  catastrophic forgetting).
- Sets up the multi-agent chapter: delayed coupling naturally leads to field-theoretic interaction pictures.

## Roadmap

1. The historical manifold and memory screen construction.
2. Memory as self-interaction: kernels, potentials, and retrieval dynamics.
3. Diagnostics and failure modes for memory-driven control.

:::{div} feynman-prose
Now we come to something that, if you think about it carefully, is rather remarkable. Up until now, everything we've done has been *local*. The agent looks at where it is right now, checks the gradient of the potential right here, and decides which way to move. It's like a hiker who can only feel the slope under their feet---very sensible, very Markovian.

But real intelligent systems don't work that way, do they? When you're solving a problem and you suddenly remember "Oh wait, I tried something like this last Tuesday and it worked beautifully"---that's not a local operation. Your past experience is reaching forward through time and grabbing you by the collar, saying "Go *this* way."

That's what this section is about: how to make the mathematics of memory. Not memory as a passive storage system---not a filing cabinet you occasionally rummage through---but memory as an active force that literally pulls the agent toward places it has succeeded before and pushes it away from places it has failed.
:::

(rb-experience-replay)=
:::{admonition} Researcher Bridge: Experience Replay as a Potential Field
:class: info
Standard Experience Replay buffers are just "bags of transitions" sampled at random. We reframe Memory as a **Self-Interaction Functional**. Past successes and failures act like physical magnets (attractive or repulsive charges) that generate a non-local force $\Psi_{\text{mem}}$. The agent does not just "sample" the past; it is literally **pulled** toward high-reward trajectories by the gradient of its own history.
:::

(sec-the-historical-manifold-and-memory-screen)=
## The Historical Manifold and Memory Screen

:::{div} feynman-prose
Before we can talk about memory as a force, we need to talk about what memory actually *is* in our geometric picture. And this is where things get interesting.

Think about an agent wandering through its latent space $\mathcal{Z}$. At every moment, it's at some position $z_t$. As time goes on, it traces out a path---a trajectory $\gamma$ through the space. Now, along this path, things happen. Sometimes the agent gets reward (good!), sometimes penalty (bad!). The trajectory isn't just a line; it's a line decorated with information about how well things went at each point.

Here's the key mental picture: imagine that every time the agent visits a location and gets reward, it leaves behind a little glowing marker---like a firefly trail. Positive reward leaves a bright warm light; negative reward leaves a cold dark spot. Over time, the agent builds up this *luminous history* across the manifold.

That's what we're going to formalize. The "memory screen" is this accumulated pattern of lights and shadows, and it's going to act like a source of gravitational (or anti-gravitational) force.
:::

**Motivation.** Sections 20--24 developed local dynamics: the geodesic SDE (Definition {prf:ref}`def-bulk-drift-continuous-flow`) evolves $z_t$ based on $\Phi_{\text{eff}}(z_t)$ and its gradient at the current position. This is a Markovian formulation---future evolution depends only on present state. However, intelligent agents demonstrably use *memory*: past experience influences current decisions through mechanisms beyond local gradients {cite}`lin1992experiencereplay,mnih2015dqn`. This section extends the geometric framework to include *non-local* contributions arising from the agent's trajectory history.

:::{prf:definition} Historical Record
:label: def-historical-record

Let $\gamma: [0, T] \to \mathcal{Z}$ be the agent's trajectory on the latent manifold $(\mathcal{Z}, G)$ over time interval $[0, T]$. The *historical record* is the pair $(\gamma, \alpha)$ where $\alpha: [0, T] \to \mathbb{R}$ is the reward flux along the trajectory (Definition {prf:ref}`def-the-reward-flux`).

*Units:* $[\gamma(t)] = [z]$, $[\alpha(t)] = \text{nat}/[s]$.

*Cross-reference:* This connects to Memory Time $t' < t$ (Definition 1.3.4).

:::

:::{div} feynman-prose
So the historical record is simply the agent's trajectory plus what happened along it---where did you go, and what reward did you get at each moment? This is the raw material of memory. Now we need to turn it into something geometric.
:::

:::{prf:definition} Memory Screen
:label: def-memory-screen

The *memory screen* is the signed measure on $\mathcal{Z}$ defined by

$$
\Xi_T := \int_0^T \alpha(t') \, \delta_{\gamma(t')} \, dt',

$$
where:
- $\delta_{\gamma(t')}$ is the Dirac measure concentrated at $\gamma(t') \in \mathcal{Z}$,
- $\alpha(t') = J_r(t')$ is the (signed) reward flux at time $t'$ (Definition {prf:ref}`def-the-reward-flux`).

*Units:* $[\Xi_T] = \text{nat}$ (total signed measure), $[\alpha] = \text{nat}/[s]$ (reward flux rate).

*Interpretation:* $\Xi_T$ encodes where the agent has been, weighted by the sign and magnitude of reward received. Positive rewards contribute positive measure (attractive memory); negative rewards contribute negative measure (repulsive memory).

*Cross-reference (Relativistic Multi-Agent):* In Chapter 29, the Memory Screen is elevated from an auxiliary construct to a **primary state variable**. The Causal Bundle $\mathcal{Z}_{\text{causal}} := \mathcal{Z}^{(N)} \times \Xi_{<t}$ restores the Markov property in relativistic multi-agent settings where finite information speed creates non-Markovian dynamics. See Definition {prf:ref}`def-causal-bundle`.

:::

:::{div} feynman-prose
Look at that definition carefully. The memory screen $\Xi_T$ is a *signed measure*---it can be positive or negative. And what it does is accumulate Dirac deltas (point masses) along the trajectory, each weighted by the reward flux $\alpha(t')$ at that moment.

Here's what that means in plain language: if you visited position $z^*$ and got positive reward there, you deposit a little positive "charge" at $z^*$. If you visited and got punished, you deposit negative charge. The memory screen is the sum total of all these deposits.

Now, why "screen"? Think of it like a projection screen in a movie theater. The agent's entire history gets projected down onto the latent space as this signed measure. All the temporal information---"I was here first, then there, then back here"---gets collapsed into a single spatial pattern. The when becomes where, weighted by how well it went.

This is not a lossy compression by accident; it's a deliberate choice. The memory screen forgets *when* things happened but remembers *where* and *how good*. That turns out to be exactly what you need to generate a conservative force field.
:::

:::{prf:remark} Connection to Holographic Persistence
:label: rem-connection-to-holographic-persistence

The memory screen $\Xi_T$ provides the mathematical realization of holographic persistence ({ref}`FAQ D.5.3 <sec-appendix-d-control-theory-system-safety>`). The measure $\Xi_T$ on $\mathcal{Z}$ acts as a "hologram" of the agent's history projected onto the latent space, from which non-local forces can be computed.

:::



(sec-the-non-local-interaction-functional)=
## The Non-Local Interaction Functional

:::{div} feynman-prose
Now comes the beautiful part. We have this memory screen---this pattern of positive and negative charges deposited across the manifold. But a bunch of point charges isn't a force field yet. We need to *smooth it out* and turn it into something that can actually push and pull the agent.

This is where we borrow a trick from physics: the heat kernel. If you've never encountered it before, here's the intuition. Imagine dropping a blob of ink into water. Initially, the ink is concentrated at one point. But heat (diffusion) spreads it out over time. The heat kernel $H_\tau(z, z')$ tells you: if you started with all the ink at point $z'$, how much ink would be at point $z$ after time $\tau$?

We're going to use this kernel to "blur" our point-charge memory into a smooth potential field. The diffusion time $\tau$ controls how far the influence spreads. Small $\tau$ means sharp, localized memory---you only feel the pull of experiences that happened very close to where you are now. Large $\tau$ means diffuse, global memory---experiences from far away can still tug at you.
:::

**Motivation.** Given the memory screen $\Xi_T$, we construct a *potential* $\Psi_{\text{mem}}(z)$ that exerts influence at the current position $z$ based on the entire historical distribution. The key mathematical object is an integral kernel that smooths and propagates the memory measure.

:::{prf:definition} Memory Kernel via Heat Equation {cite}`grigoryan2009heat,rosenberg1997laplacian`
:label: def-memory-kernel-via-heat-equation

The canonical memory kernel is the *Heat Kernel* $H_\tau(z, z')$ on $(\mathcal{Z}, G)$, defined as the fundamental solution to the heat equation:

$$
(\partial_\tau - \Delta_G) H_\tau(z, z') = 0, \quad H_0(z, z') = \delta(z - z'),

$$
where:
- $\tau > 0$ is the *diffusion time* (memory smoothing scale),
- $\Delta_G = G^{ij}\nabla_i\nabla_j$ is the Laplace-Beltrami operator on $(\mathcal{Z}, G)$ (Definition 2.5.3).

*Units:* $[H_\tau] = [z]^{-d}$ (probability density), $[\tau] = [z]^2$ (diffusion time in geometric units).

*Interpretation:* $H_\tau(z, z')$ measures how much influence a memory at $z'$ has on the current position $z$ after diffusion time $\tau$. Larger $\tau$ yields smoother, more diffuse memory influence. For compact manifolds, $H_\tau$ admits an eigenfunction expansion; for non-compact manifolds with bounded geometry, Gaussian upper bounds hold {cite}`grigoryan2009heat`.

:::

:::{div} feynman-prose
Notice something important: we're using the *geometry* of the latent space here. The Laplace-Beltrami operator $\Delta_G$ knows about the metric $G$. This means memories don't spread uniformly in all directions---they spread along the natural geometry of the space. If two regions are close in geodesic distance (even if they look far apart in Euclidean coordinates), memories will flow between them easily.

This is exactly right! The metric $G$ encodes what it means for states to be "similar" in a control-relevant sense. Memory influence should follow the same notion of similarity.
:::

:::{prf:definition} Memory Potential
:label: def-memory-potential

The *memory potential* is defined by

$$
\Psi_{\text{mem}}(z) := -\int_{\mathcal{Z}} H_\tau(z, z') \, d\Xi_T(z').

$$
Expanding using Definition {prf:ref}`def-memory-screen`:

$$
\Psi_{\text{mem}}(z) = -\int_0^T \alpha(t') H_\tau(z, \gamma(t')) \, dt'.

$$
*Units:* $[\Psi_{\text{mem}}] = \text{nat}$.

*Interpretation:* The memory potential is the convolution of the heat kernel with the signed reward-weighted trajectory measure. Since $\Xi_T$ is a signed measure:
- Near high-reward past positions ($\alpha > 0$): $\Psi_{\text{mem}} < 0$, creating a potential well. The force $-\nabla_G \Psi_{\text{mem}}$ points toward the memory (attractive).
- Near high-penalty past positions ($\alpha < 0$): $\Psi_{\text{mem}} > 0$, creating a potential barrier. The force $-\nabla_G \Psi_{\text{mem}}$ points away from the memory (repulsive).

The sign convention ensures that the drift term inside $\mathcal{M}_{\text{curl}}\!\left(-G^{-1}\nabla \Psi_{\text{mem}}\right)$ moves toward rewarding experiences and away from penalizing ones.

:::

:::{div} feynman-prose
Let me make sure you've got the physics right in your head. The memory potential $\Psi_{\text{mem}}$ is like an electrostatic potential, and the reward-weighted trajectory acts like a distribution of electric charges.

Positive reward at a past location? That's positive charge. But wait---with the minus sign in the definition, positive charge creates *negative* potential. And $-\nabla\Psi_{\text{mem}}$ points toward more negative potential. So the force points *toward* high-reward locations. The agent is attracted to where things went well.

Negative reward? Negative charge. Creates positive potential. Force points away. The agent is repelled from where things went badly.

It's like having a bunch of tiny magnets scattered across the manifold, some attracting and some repelling, all adding up to create a force field that guides the agent's decisions. The strength falls off with distance (through the heat kernel), so nearby memories matter more than distant ones.
:::

(pi-heat-kernel)=
::::{admonition} Physics Isomorphism: Heat Kernel
:class: note

**In Physics:** The heat kernel $H_t(x, y)$ is the fundamental solution of the heat equation $\partial_t u = \Delta u$, satisfying $H_t(x, \cdot) \to \delta_x$ as $t \to 0$. On a Riemannian manifold, it encodes diffusion and satisfies $H_t(x, y) \sim (4\pi t)^{-d/2}\exp(-d^2(x,y)/4t)$ for small $t$ {cite}`berline1992heat,grigoryan2009heat`.

**In Implementation:** The memory potential uses heat kernel convolution (Definition {prf:ref}`def-memory-potential`):

$$
\Psi_{\text{mem}}(z) = -\int_0^T \alpha(t') H_\tau(z, \gamma(t'))\, dt'

$$
where $H_\tau$ is the heat kernel on $(\mathcal{Z}, G)$.

**Correspondence Table:**
| Heat Equation Theory | Agent (Non-Local Memory) |
|:---------------------|:-------------------------|
| Heat kernel $H_t(x, y)$ | Memory diffusion kernel |
| Diffusion time $t$ | Memory timescale $\tau$ |
| Heat source | Past reward-weighted positions |
| Temperature evolution | Belief spread over time |
| Short-time asymptotics | Geodesic distance dominance |

**Connection:** The Matérn kernel $K_\nu \propto (-\Delta_G + \kappa^2)^{-\nu}$ generalizes the heat kernel; for $\nu = 1$, it recovers the screened Poisson Green's function ({ref}`sec-the-bulk-potential-screened-poisson-equation`).
::::

:::{prf:proposition} Kernel Alternatives {cite}`rasmussen2006gp`
:label: prop-kernel-alternatives

Alternative kernels may be used depending on application requirements:

1. **Gaussian (RBF) Kernel:**

   $$
   K_{\text{Gauss}}(z, z') := \exp\left(-\frac{d_G(z, z')^2}{2\ell^2}\right),

   $$
   where $d_G$ is the geodesic distance and $\ell > 0$ is the length scale. This provides fast (exponential) decay, suitable for short-range memory effects.

2. **Matérn Kernel:**

   $$
   K_{\nu}(z, z') \propto (-\Delta_G + \kappa^2)^{-\nu}\delta(z - z'),

   $$
   where $\nu > 0$ is the smoothness parameter and $\kappa > 0$ is the inverse correlation length. For $\nu = 1$, this recovers the Green's function $G_\kappa$ from {ref}`sec-the-bulk-potential-screened-poisson-equation`. The Matérn kernel has polynomial (rather than exponential) tails, providing longer-range correlations. See {cite}`rasmussen2006gp` Chapter 4 for the Euclidean case.

*Cross-reference:* The Matern kernel with $\nu = 1$ coincides with the screened Poisson Green's function (Definition {prf:ref}`prop-green-s-function-decay`), establishing a direct connection between memory effects and value propagation.

:::

:::{div} feynman-prose
Why do we have multiple kernel choices? Because different problems have different memory structures.

The heat kernel (Gaussian decay) says: "Recent experiences matter, but old ones fade away exponentially fast." Good for fast-changing environments where yesterday's success might be today's trap.

The Matern kernel (polynomial tails) says: "Old experiences still have some pull, just weaker." Good for stable environments where what worked a thousand steps ago probably still works.

The key insight is that the choice of kernel is a modeling decision about *how memory should decay with distance*. There's no universally correct answer---it depends on the structure of your problem.
:::

:::{prf:theorem} Non-Markovian Nature of Memory
:label: thm-non-markovian-nature-of-memory

The force field $-\nabla_G \Psi_{\text{mem}}$ violates the Markov property.

*Proof.* By Definition {prf:ref}`def-memory-potential`, $\Psi_{\text{mem}}(z_t)$ depends on $\Xi_T$, which contains $\gamma(t')$ for all $t' < t$. Therefore, $\nabla_G \Psi_{\text{mem}}(z_t)$ depends on the entire trajectory history $\{\gamma(t')\}_{t' \in [0,t)}$, not merely on $z_t$. This violates the Markov property $P(z_{t+\delta} | z_t, \{z_s\}_{s<t}) = P(z_{t+\delta} | z_t)$. $\square$

*Remark (State Augmentation):* The non-Markovian character is essential for capturing genuine memory effects. The system state must be *augmented* to include $\Xi_T$ (or a sufficient statistic thereof) to recover a Markovian description in an extended state space.

*Remark (Computational Complexity):* Naively, evaluating $\Psi_{\text{mem}}(z)$ requires $O(T)$ kernel evaluations where $T$ is the trajectory length. For long histories, approximations are necessary: (i) truncate to recent history, (ii) subsample the trajectory, (iii) use inducing points {cite}`rasmussen2006gp`, or (iv) maintain a running kernel density estimate.

:::

:::{div} feynman-prose
This theorem is saying something profound, so let me make sure it's clear. A Markov process is one where the future depends only on the present---you can throw away all your history and still make optimal predictions. That's a beautiful mathematical property, and it makes life much simpler.

But memory, by definition, violates this. The force the agent feels *right now* depends on where it's been *in the past*. You can't throw away the history. The present position $z_t$ is not a sufficient statistic for predicting what force the agent will experience.

Now, this isn't as catastrophic as it sounds. The theorem also points to the solution: if you *augment* the state to include the memory screen $\Xi_T$ (or some approximation of it), you get back a Markov process in the larger state space. The agent's state isn't just "where am I?" but "where am I, and what does my memory map look like?"

This is the standard trick for handling non-Markovian dynamics: expand your notion of state until it becomes Markovian again.
:::

(sec-memory-augmented-equations-of-motion)=
## Memory-Augmented Equations of Motion

:::{div} feynman-prose
Now we get to put everything together. We had the geodesic SDE from before---the equation that tells the agent how to move through latent space based on local potentials, policy, and noise. Now we're going to add one more term: the memory force.

The beautiful thing is that it fits in so naturally. The memory potential $\Psi_{\text{mem}}$ just *adds* to the effective potential $\Phi_{\text{eff}}$. Physics is additive---if you have two sources of force, you just add them. The agent feels the gradient of the total potential, which now includes contributions from both the local value landscape and the non-local memory field.
:::

**Motivation.** We now extend the geodesic SDE (Definition {prf:ref}`def-bulk-drift-continuous-flow`) to include the memory-induced force $-\nabla_G \Psi_{\text{mem}}$.

:::{prf:definition} Memory-Augmented Geodesic SDE
:label: def-memory-augmented-geodesic-sde

The memory-augmented dynamics on $(\mathcal{Z}, G)$ are:

$$
dz^k = \left[\mathcal{M}_{\text{curl}}\right]^k{}_{j}\left(-G^{j\ell}\partial_\ell\bigl(\Phi_{\text{eff}} + \Psi_{\text{mem}}\bigr) + u_\pi^j\right) ds - \Gamma^k_{ij}\dot{z}^i\dot{z}^j\,ds + \sqrt{2T_c}\,(G^{-1/2})^{kj}\,dW^j_s,

$$
where:
- $\Phi_{\text{eff}}$ is the effective potential (Definition {prf:ref}`def-effective-potential`),
- $\Psi_{\text{mem}}$ is the memory potential (Definition {prf:ref}`def-memory-potential`),
- $\Gamma^k_{ij}$ are the Christoffel symbols of $G$ (Definition 2.5.1),
- $u_\pi^k$ is the policy control field (Definition {prf:ref}`def-the-control-field`),
- $T_c$ is the cognitive temperature ({prf:ref}`def-cognitive-temperature`, {ref}`sec-the-geodesic-baoab-integrator`),
- $W^j_s$ is a standard Wiener process,
- $\mathcal{M}_{\text{curl}} := (I - \beta_{\text{curl}} G^{-1}\mathcal{F})^{-1}$ is the curl-corrected mobility.

*Cross-reference:* Definition {prf:ref}`def-bulk-drift-continuous-flow`.

*Units:* All terms have units $[z]/[s]$.

:::

:::{div} feynman-prose
Look at that equation. It's the same structure as before---drift from potential gradients, geodesic correction (the Christoffel symbols), policy control, and noise. The only change is that the potential is now $\Phi_{\text{eff}} + \Psi_{\text{mem}}$ instead of just $\Phi_{\text{eff}}$.

This is the power of the geometric framework. Memory doesn't require a whole new theory; it slots right into the existing equations as an additional potential. The agent doesn't have to "decide to use memory"---memory automatically exerts its influence through the same gradient-following mechanism that drives all the other dynamics.
:::

:::{prf:lemma} Virtual Work of Recall
:label: lem-virtual-work-of-recall

The infinitesimal work performed by the memory force during displacement $dz$ is:

$$
dW_{\text{mem}} := \langle -\nabla_G \Psi_{\text{mem}}, dz \rangle_G = -G_{kj}\,G^{k\ell}\partial_\ell \Psi_{\text{mem}}\, dz^j = -\partial_j \Psi_{\text{mem}}\, dz^j.

$$
*Units:* $[dW_{\text{mem}}] = \text{nat}$.

*Interpretation:* When the agent moves toward regions of low $\Psi_{\text{mem}}$ (attractive memory, i.e., $d\Psi_{\text{mem}} < 0$), positive work $dW_{\text{mem}} > 0$ is extracted from the memory field. This corresponds to "reward from recall"---revisiting previously successful states.

:::

:::{div} feynman-prose
The "virtual work of recall" is a lovely concept. When you remember a past success and move toward recreating it, you're extracting work from the memory field. It's as if your memories are batteries storing potential energy, and using them releases that energy to help drive your current behavior.

Conversely, moving toward bad memories costs work---the memory field resists. This is exactly what you'd want: the mathematics encodes "seek experiences like past successes" and "avoid experiences like past failures" as energetic principles.
:::

:::{prf:theorem} Memory-Induced Barrier Crossing
:label: thm-memory-induced-barrier-crossing

Let $z_t$ be the current position and suppose there exists a past time $t^* < t$ with $z^* := \gamma(t^*)$ such that:
1. $d_G(z_t, z^*) < \ell_{\text{mem}}$ for some memory influence radius $\ell_{\text{mem}}$,
2. $|\alpha(t^*)|$ is large (strong reward signal at time $t^*$).

Then the memory gradient $\|\nabla_G \Psi_{\text{mem}}\|_G$ can exceed the local barrier gradient $\|\nabla_G \Phi_{\text{eff}}\|_G$, enabling transitions that would be forbidden under purely local dynamics.

*Proof sketch.* By Definition {prf:ref}`def-memory-potential` and the concentration of $H_\tau$ near the diagonal for small $\tau$:

$$
\|\nabla_G \Psi_{\text{mem}}(z_t)\|_G \approx |\alpha(t^*)| \cdot \|\nabla_G H_\tau(z_t, z^*)\|_G.

$$
For $d_G(z_t, z^*) \sim O(\sqrt{\tau})$, the gradient $\|\nabla_G H_\tau\|_G \sim O(\tau^{-(d+1)/2})$ can be made arbitrarily large by choosing small $\tau$. If $|\alpha(t^*)|$ is sufficiently large, this dominates $\|\nabla_G \Phi_{\text{eff}}\|_G$. $\square$

*Cross-reference:* BarrierGap diagnostic ({ref}`sec-4-limits-barriers-the-limits-of-control`).

*Interpretation:* Strong memories can "pull" the agent across local energy barriers, providing a mechanism for experience-guided exploration that transcends gradient-based planning.

:::

:::{div} feynman-prose
This theorem is saying something genuinely important about the power of memory. Without memory, the agent is trapped by local energy barriers. If there's a hill between where you are and where you want to be, you can't get there---the gradient just pushes you back down.

But with memory, if you've been on the other side of that hill before and it was good, the memory field reaches across the barrier and *pulls*. For strong enough memories close enough to your current position, this pull can overcome the local barrier.

Think about what this means for exploration. Pure gradient-following is myopic---you only see the immediate landscape. Memory lets you see through walls, in a sense. A strong memory of success acts like a beacon, pulling you toward it even when the local landscape says "go away."

This is how experience-guided exploration works. You don't just wander randomly hoping to stumble on good states. Your past successes reach out and guide you back.
:::

::::{admonition} Connection to RL #20: Experience Replay as Degenerate Non-Local Memory
:class: note
:name: conn-rl-20
**The General Law (Fragile Agent):**
Trajectory history induces a **Memory Potential** via heat-kernel convolution:

$$
\Psi_{\text{mem}}(z) = -\int_0^T \alpha(t') H_\tau(z, \gamma(t'))\, dt'

$$
where $H_\tau$ is the heat kernel on $(\mathcal{Z}, G)$ and $\alpha(t')$ is the reward flux at past times.

**The Degenerate Limit:**
Replace geometric kernel with uniform sampling. Ignore metric structure ($G \to I$).

**The Special Case (Standard RL):**

$$
\mathcal{D} = \{(s_t, a_t, r_t, s_{t+1})\}, \quad \text{sample } \sim \text{Uniform}(\mathcal{D})

$$
This recovers **Experience Replay** {cite}`lin1992experience,mnih2015humanlevel`.

**What the generalization offers:**
- **Geometric memory:** Distances measured in $d_G$, not Euclidean; nearby trajectories interact more strongly
- **Reward-signed forces:** Positive rewards attract (revisit success); negative repel (avoid failure)
- **Heat-kernel smoothing:** Memory influence decays with diffusion time $\tau$
- **Barrier crossing:** Strong memories can pull the agent across local energy barriers
::::



(sec-wfr-dynamics-with-memory-sources)=
## WFR Dynamics with Memory Sources

:::{div} feynman-prose
So far we've been thinking about a single agent following a trajectory. But remember from Section 20 that we can also think about *beliefs*---probability distributions over where the agent might be. The Wasserstein-Fisher-Rao (WFR) framework describes how these distributions evolve.

What happens when we add memory to the WFR picture? Something quite remarkable: memory acts as a *source term* in the continuity equation. Belief mass can be created where attractive memories live and destroyed where repulsive memories live. It's as if the memory screen is shining spotlights that make certain regions of the space more "real" (higher probability) and casting shadows that make other regions less real.
:::

**Motivation.** Lifting to measure space via the Wasserstein-Fisher-Rao framework ({prf:ref}`def-the-wfr-action`, {ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`) {cite}`chizat2018wfr`, we obtain a reaction-diffusion PDE incorporating memory.

:::{prf:definition} Memory-Augmented Reaction-Diffusion
:label: def-memory-augmented-reaction-diffusion

The WFR dynamics with memory are:

$$
\partial_s \rho + \nabla \cdot (\rho \mathbf{v}) = \rho \left(\frac{\Phi_{\text{eff}} + \Psi_{\text{mem}} - \bar{\Phi}_{\text{aug}}}{T_c}\right),

$$
where:
- $\rho(z, s)$ is the belief density,
- $\mathbf{v} = \mathcal{M}_{\text{curl}}\!\left(-G^{-1}\nabla(\Phi_{\text{eff}} + \Psi_{\text{mem}}) + u_\pi\right)$ is the curl-corrected drift,
- $\bar{\Phi}_{\text{aug}} = \int_{\mathcal{Z}} (\Phi_{\text{eff}} + \Psi_{\text{mem}}) \rho \, d\mu_G$ is the mean augmented potential.

*Cross-reference:* Definition {prf:ref}`def-the-wfr-action`, Theorem {prf:ref}`thm-wfr-consistency-value-creates-mass`.

*Units:* $[\partial_s \rho] = [z]^{-d}/[s]$, all terms balance.

:::
:::{prf:proposition} Mass Creation from Experience
:label: prop-mass-creation-from-experience

The memory contribution to the reaction term is:

$$
r_{\text{mem}}(z) := \frac{\rho(z)(\Psi_{\text{mem}}(z) - \bar{\Psi}_{\text{mem}})}{T_c},

$$
where $\bar{\Psi}_{\text{mem}} = \int_{\mathcal{Z}} \Psi_{\text{mem}} \rho \, d\mu_G$.

*Interpretation:* Belief mass is created where $\Psi_{\text{mem}} < \bar{\Psi}_{\text{mem}}$ (attractive memory) and destroyed where $\Psi_{\text{mem}} > \bar{\Psi}_{\text{mem}}$ (repulsive memory). This acts as a *virtual source* that redistributes probability toward remembered high-reward regions, even when local dynamics (via $\Phi_{\text{eff}}$) do not support such transitions.

:::

:::{div} feynman-prose
Here's what this means in plain language. Normally, probability flows around like an incompressible fluid---what leaves one place must arrive somewhere else. But the reaction term allows mass creation and destruction. Memory makes probability *appear* near good experiences and *disappear* near bad ones.

This is a different mechanism from transport. Transport moves existing probability around. Reaction creates and destroys it. With memory, the agent's beliefs can spontaneously shift toward remembered successes without having to continuously flow there through the space.

In computational terms, this is like teleportation. Instead of walking from A to B through all the intermediate states, memory lets you just... appear at B, if B was a strongly positive experience. The strength of this effect is proportional to how much better (or worse) the memory potential is at that location compared to the average.
:::

(sec-stability-analysis-and-diagnostic)=
## Stability Analysis and Diagnostic

:::{div} feynman-prose
Memory is powerful, but power is dangerous. If memory pulls too hard, the agent becomes a slave to its past---it will keep trying to recreate old successes even when the world has changed and those strategies no longer work. That's overfitting to history.

On the other hand, if memory is too weak, the agent forgets what worked and keeps making the same mistakes over and over. That's catastrophic forgetting.

The healthy regime is somewhere in between: memory should inform but not dominate. The question is: how do we know if we're in the healthy regime? That's what this section is about---a diagnostic that tells you whether memory and local dynamics are properly balanced.
:::

**Motivation.** Non-local memory introduces a potential source of instability: if memory forces dominate local dynamics, the agent may overfit to history and fail to adapt to environmental changes. Conversely, if memory is too weak, the agent exhibits catastrophic forgetting.

:::{prf:definition} Non-Locality Ratio
:label: def-non-locality-ratio

The *non-locality ratio* at position $z$ is:

$$
\Omega_{\text{mem}}(z) := \frac{\|\nabla_G \Psi_{\text{mem}}(z)\|_G}{\|\nabla_G \Phi_{\text{eff}}(z)\|_G + \epsilon},

$$
where $\epsilon > 0$ is a regularization constant preventing division by zero.

*Units:* $[\Omega_{\text{mem}}] = \text{dimensionless}$.

**Heuristic 27.5.2 (Homeostatic Bound on Memory).** For stable operation, the non-locality ratio should satisfy:

$$
\Omega_{\text{mem}} \in [\Omega_{\min}, \Omega_{\max}],

$$
with empirically recommended bounds $\Omega_{\min} \approx 0.01$, $\Omega_{\max} \approx 10$. These bounds are task-dependent and should be tuned based on the environment's stationarity.

*Boundary cases:*
- $\Omega_{\text{mem}} \to 0$: Pure Markovian dynamics; agent exhibits catastrophic forgetting.
- $\Omega_{\text{mem}} \to \infty$: Pure memory-driven dynamics; agent overfits to historical experience and fails to respond to current environmental gradients.

*Cross-reference:* The Governor ({ref}`sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller`) can regulate $\Omega_{\text{mem}}$ by adjusting the memory smoothing scale $\tau$ or the reward flux weighting in $\alpha(t')$.

:::

:::{div} feynman-prose
The non-locality ratio $\Omega_{\text{mem}}$ is beautifully simple: it's just the ratio of how hard memory is pulling versus how hard the local potential is pulling. If $\Omega_{\text{mem}} \approx 1$, they're about equally matched. If $\Omega_{\text{mem}} \gg 1$, memory dominates. If $\Omega_{\text{mem}} \ll 1$, local dynamics dominate.

The heuristic bounds make operational sense. You probably want $\Omega_{\text{mem}}$ somewhere between 0.01 and 10---memory should be noticeable but not overwhelming. The exact bounds depend on how stationary your environment is. In a very stable environment where the past is a good guide to the future, you can afford larger $\Omega_{\text{mem}}$. In a rapidly changing environment, you want smaller $\Omega_{\text{mem}}$ so the agent stays responsive to current conditions.
:::

(node-43)=
**Node 43: MemoryBalanceCheck**

| **#**  | **Name**               | **Component**     | **Type**              | **Interpretation**              | **Proxy**                                                                                                 | **Cost**               |
|--------|------------------------|-------------------|-----------------------|---------------------------------|-----------------------------------------------------------------------------------------------------------|------------------------|
| **43** | **MemoryBalanceCheck** | **Memory Screen** | **Non-Local Balance** | Is memory contribution bounded? | $\Omega_{\text{mem}} = \lVert\nabla_G\Psi_{\text{mem}}\rVert_G / \lVert\nabla_G\Phi_{\text{eff}}\rVert_G$ | $O(\lvert\Xi_T\rvert)$ |

**Trigger conditions:**
- $\Omega_{\text{mem}} < \Omega_{\min}$: Memory underutilized; increase $\alpha$ weighting or decrease $\tau$.
- $\Omega_{\text{mem}} > \Omega_{\max}$: Memory dominates; increase $\tau$ to smooth memory influence or decay old experiences.
- Persistent imbalance: Re-examine memory kernel choice or trajectory sampling strategy.

**Cross-references:** {ref}`sec-diagnostics-stability-checks` (Sieve Diagnostic Nodes), {ref}`sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller` (Governor regulation), Node 42 (GovernorStabilityCheck).

:::



(sec-summary-memory-as-non-local-interface)=
## Summary: Memory as Non-Local Interface

:::{div} feynman-prose
Let's step back and see what we've built. The Fragile Agent has four main pillars---Perception, Action, Value, and Memory. The first three are all *local*: they depend only on what's happening right here, right now. Perception clamps your position based on what you observe. Action clamps your momentum based on what you do. Value provides a local source term based on reward.

Memory is different. Memory is the agent talking to itself across time. It's non-local in the most fundamental sense: what you feel now depends on what you experienced in the past, potentially long ago and far away in state space. The mathematical signature of this is the Fredholm integral operator---an integral over the entire trajectory, not just a differential operator at a point.

The table below makes this distinction crisp.
:::

**Table 27.6.1 (Pillar Locality Comparison).**

| Pillar     | Operator            | Geometric Role                   | Locality      |
|------------|---------------------|----------------------------------|---------------|
| Perception | $E_\phi$            | Dirichlet BC (position clamping) | Local         |
| Action     | $D_A$               | Neumann BC (flux clamping)       | Local         |
| Value      | $\Phi_{\text{eff}}$ | Source BC (Helmholtz solution)   | Local         |
| **Memory** | $\Psi_{\text{mem}}$ | **Fredholm integral operator**   | **Non-local** |

*Key insight:* Memory introduces the first genuinely non-local contribution to the agent dynamics. While perception, action, and value all depend on local data (position, flux, source at $z$), memory integrates information over the entire trajectory history via the kernel $H_\tau$.

**Table 27.6.2 (Memory Kernel Comparison).**

| Kernel         | Asymptotic Form                         | Decay Rate              | Use Case                                  |
|----------------|-----------------------------------------|-------------------------|-------------------------------------------|
| Heat $H_\tau$  | $(4\pi\tau)^{-d/2}\exp(-d_G^2/4\tau)$   | Gaussian                | Default; smooth diffusive influence       |
| Gaussian/RBF   | $\exp(-d_G^2/2\ell^2)$                  | Exponential             | Short-range memory; fast computation      |
| Matérn $K_\nu$ | $d_G^{\nu-d/2} K_{\nu-d/2}(\kappa d_G)$ | Polynomial $\times$ exp | Long-range; connects to value propagation |

*Note:* $K_{\nu}$ denotes the modified Bessel function of the second kind. For the Matérn kernel on curved manifolds, the formula is approximate; exact expressions require spectral methods.

**Summary.** This section introduced non-local memory as a self-interaction functional, extending the Markovian dynamics of Sections 20–24. The memory screen $\Xi_T$ (Definition {prf:ref}`def-memory-screen`) encodes reward-weighted trajectory history; the memory potential $\Psi_{\text{mem}}$ (Definition {prf:ref}`def-memory-potential`) converts this into a force field via heat kernel convolution; and the Non-Locality Ratio $\Omega_{\text{mem}}$ (Definition {prf:ref}`def-non-locality-ratio`) provides a diagnostic for balancing memory against local gradients. Node 43 (MemoryBalanceCheck) monitors this ratio during training.



(sec-section-hyperbolic-active-retrieval-geodesic-search-and-semantic-pull-back)=
## Hyperbolic Active Retrieval: Geodesic Search and Semantic Pull-Back

:::{div} feynman-prose
Section 27 was about the agent remembering its own past. Now we turn to something different but closely related: the agent reaching out to *external* knowledge. Think of this as the difference between "I remember doing this before" and "Let me look that up."

In modern AI systems, this is called Retrieval-Augmented Generation (RAG)---the system queries a knowledge base and incorporates what it finds into its reasoning. But how do we fit this into our geometric framework? Here's the beautiful insight: if both the agent's internal representations and the external knowledge base use compatible embeddings, then retrieval is just another kind of non-local interaction. Instead of being pulled by your own past, you're being pulled by relevant facts in the knowledge base.

The key challenge is what we call the "texture firewall problem." External documents contain lots of specific details---exact wordings, formatting, style. You want to *use* that information for generation, but you don't want it to corrupt your *reasoning*. The solution is to carefully separate what goes to the control loop (just the semantic content) from what goes to the decoder (full details including texture).
:::

(rb-retrieval-augmented)=
:::{admonition} Researcher Bridge: Retrieval-Augmented Control
:class: info
If you know episodic control or retrieval-augmented generation, this is the geometric version: retrieval is a geodesic search in a shared embedding space. The firewall ensures retrieved texture does not leak into policy decisions.
:::

*Cross-references:*
- {ref}`sec-radial-generation-entropic-drift-and-policy-control` (Poincare metric).
- {ref}`sec-the-equations-of-motion-geodesic-jump-diffusion` (Equations of Motion).
- {ref}`sec-section-non-local-memory-as-self-interaction-functional` (Memory potential).
- {ref}`sec-tier-the-attentive-atlas` (Atlas architecture).
- {ref}`sec-conditional-independence-and-sufficiency` (Macro closure).

**Motivation.** While {ref}`sec-section-non-local-memory-as-self-interaction-functional` treated memory as self-interaction—retrieval from the agent's own trajectory—this section addresses *external* retrieval from knowledge bases, embedding indices, and document stores. The central observation is that the Poincare disk geometry introduced in {ref}`sec-radial-generation-entropic-drift-and-policy-control` applies equally to both internal latent representations and external knowledge embeddings. This isomorphism enables principled Retrieval-Augmented Generation (RAG) as geodesic search on a shared hyperbolic manifold.

The key challenge is the **texture firewall problem**: external documents contain high-frequency texture ($z_{\text{tex}}$) that must be delivered to the decoder but excluded from the control loop to prevent Mode T.C (Labyrinthine Overfitting). We solve this by extending the existing TextureFirewallCheck (Node 29) to external retrieval, ensuring that only bulk coordinates $(K, z_n)$ influence policy gradients.

(sec-the-isomorphism-of-semantic-manifolds)=
## The Isomorphism of Semantic Manifolds

:::{div} feynman-prose
Here's a remarkable fact about modern embedding systems: when you train models on language, images, or any other modality, they end up learning similar geometric structures. Two different embedding models, trained on similar data, will organize concepts in compatible ways. "Cat" and "dog" will be close together (both animals), and "cat" and "democracy" will be far apart, regardless of which model you use.

This isn't magic---it's because the models are learning the same underlying semantic relationships from similar data. And it means we can treat the agent's internal representations and external knowledge bases as living in *the same* geometric space, or at least in spaces that are related by a smooth mapping.

The formal statement of this is the "Metric Isometry" axiom: there exists a distance-preserving map between internal and external representations. When this holds, retrieval becomes a geometric operation---finding nearby points in a shared space.
:::

:::{prf:definition} External Knowledge Manifold
:label: def-external-knowledge-manifold

Let $\mathcal{Z}_{\text{ext}}$ denote the external knowledge manifold equipped with metric $G_{\text{ext}}$, structured as a fiber bundle:

$$
\mathcal{Z}_{\text{ext}} = \mathcal{K} \times \mathcal{Z}_n \times \mathcal{Z}_{\text{tex}},

$$
where $\mathcal{K}$ is the macro-concept space, $\mathcal{Z}_n$ the nuisance coordinates, and $\mathcal{Z}_{\text{tex}}$ the texture fiber.

*Units:* $[G_{\text{ext},ij}] = [z]^{-2}$ (matching the internal metric).

*Cross-reference:* This decomposition mirrors {ref}`sec-conditional-independence-and-sufficiency`'s latent structure $(K, z_n, z_{\text{tex}})$ and {ref}`sec-tier-the-attentive-atlas`'s Atlas architecture.

:::
:::{prf:axiom} Metric Isometry
:label: ax-metric-isometry

There exists a canonical isometry $\Phi: \mathcal{Z}_{\text{int}} \to \mathcal{Z}_{\text{ext}}$ such that for all $z, z' \in \mathcal{Z}_{\text{int}}$:

$$
d_{G_{\text{int}}}(z, z') = d_{G_{\text{ext}}}(\Phi(z), \Phi(z')),

$$
where both manifolds carry the Poincare metric (Definition {prf:ref}`def-hyperbolic-volume-growth`):

$$
G_{ij}(z) = \frac{4\delta_{ij}}{(1 - \|z\|^2)^2}.

$$
*Interpretation:* The isometry axiom asserts that embedding models trained on shared semantic corpora induce compatible distance structures. This is the mathematical foundation for cross-modal retrieval.

:::
:::{prf:definition} Knowledge Atom
:label: def-knowledge-atom

A *knowledge atom* is a triple $\xi = (K, z_n, z_{\text{tex}}) \in \mathcal{Z}_{\text{ext}}$ where:
- $K \in \mathcal{K}$: macro-concept (topic, entity class, logical category)
- $z_n \in \mathcal{Z}_n$: nuisance coordinates (style, formatting, source metadata)
- $z_{\text{tex}} \in \mathcal{Z}_{\text{tex}}$: high-frequency texture (specific wording, surface form)

*Cross-reference:* Compare {ref}`sec-conditional-independence-and-sufficiency`'s decomposition. The macro closure mechanism (Definition 2.8.1) applies equally to external atoms.

:::

:::{div} feynman-prose
A knowledge atom is just the external version of the agent's internal state decomposition. Every piece of knowledge has a topic ($K$), some context ($z_n$), and specific surface details ($z_{\text{tex}}$). When you look up "the capital of France," the topic is "geography/capitals," the context might be "European politics," and the texture is the specific string "Paris" with whatever formatting the source uses.

The crucial insight is that for *reasoning*, you only need $K$ and $z_n$. The texture is for *output*. If the agent starts making decisions based on whether the retrieved document used "Paris" or "paris" or "PARIS," something has gone wrong.
:::

(sec-geodesic-search-in-hyperbolic-space)=
## Geodesic Search in Hyperbolic Space

:::{div} feynman-prose
Why hyperbolic space? Remember from Section 21 that the Poincare disk is the natural geometry for hierarchical data. Abstract concepts live near the center; specific instances live near the boundary. The volume of the space grows exponentially as you move outward, which perfectly captures how the number of specific facts explodes as you get more precise.

Retrieval in this geometry is *geodesic search*: given your current query position, find the nearest knowledge atoms. But "nearest" means geodesic distance, not Euclidean distance. Two facts that are both highly specific (near the boundary) might be close in Euclidean terms but far in geodesic terms if they're in different semantic branches.
:::

:::{prf:definition} Hyperbolic Geodesic Distance
:label: def-hyperbolic-geodesic-distance

For points $z, \xi \in \mathbb{D}^d$ (the Poincare disk), the geodesic distance is:

$$
d_{\mathbb{D}}(z, \xi) = \operatorname{acosh}\left(1 + \frac{2\|z - \xi\|^2}{(1 - \|z\|^2)(1 - \|\xi\|^2)}\right).

$$
*Units:* $[d_{\mathbb{D}}] = [z]$ (dimensionless in Poincare coordinates).

*Cross-reference:* This is the distance function induced by the Poincare metric $G_{ij}$ (Definition {prf:ref}`def-hyperbolic-volume-growth`). See also Definition {prf:ref}`prop-isotropic-radial-expansion` for the hyperbolic potential $U(z) = -2\operatorname{artanh}(\|z\|)$.

:::
:::{prf:definition} Retrieval Measure via Geodesic Functional
:label: def-retrieval-measure-via-geodesic-functional

Given a query position $z \in \mathcal{Z}_{\text{int}}$ and archive prior $\mu_{\mathcal{E}} \in \mathcal{P}(\mathcal{Z}_{\text{ext}})$, the *retrieval measure* is:

$$
\nu_\omega = \arg\min_{\nu \in \mathcal{P}(\mathcal{Z}_{\text{ext}})} \left\{ \int d_{\mathbb{D}}(z, \xi) \, d\nu(\xi) + T_{\text{ret}} D_{\text{KL}}(\nu \| \mu_{\mathcal{E}}) \right\},

$$
where $T_{\text{ret}} > 0$ is the *retrieval temperature*.

*Units:* $[T_{\text{ret}}] = \text{nat}$.

*Interpretation:* This variational problem balances semantic proximity (first term) against prior plausibility (KL term). At $T_{\text{ret}} \to 0$, retrieval concentrates on the nearest neighbor; at $T_{\text{ret}} \to \infty$, it reverts to the archive prior.

:::
:::{prf:proposition} Exponential Complexity of Specificity
:label: prop-exponential-complexity-of-specificity

The volume of a geodesic ball in the Poincare disk grows exponentially with radius:

$$
\text{Vol}(B_r(z)) \sim \sinh^{d-1}(r) \sim \frac{1}{2^{d-1}} e^{(d-1)r} \quad \text{as } r \to \infty.

$$
*Proof sketch:* The hyperbolic metric has constant negative curvature $\kappa = -1$. Standard volume comparison (Bishop-Gromov) yields exponential growth. $\square$

*Interpretation:* As the agent descends toward the boundary (increasing semantic specificity), the number of accessible knowledge atoms grows exponentially. This captures the combinatorial explosion of specific facts relative to abstract concepts---compare TopoEncoder hierarchy ({ref}`sec-supervised-topology-semantic-potentials-and-metric-segmentation`).

:::

:::{div} feynman-prose
This exponential volume growth is profound. At the abstract level ("animals"), there's a small number of relevant facts. At the specific level ("the individual tiger named Shere Khan in Kipling's *Jungle Book*"), there are vastly more. Hyperbolic geometry captures this hierarchy naturally: the further you go from the center, the more space there is, and the more facts can fit.

This also explains why retrieval gets harder as you get more specific. At high abstraction, there are few candidates to search through. At high specificity, the search space explodes. The geodesic distance formula reflects this: it's harder (larger distance) to distinguish between two specific facts in different domains than between two abstract concepts.
:::

(sec-the-retrieval-texture-firewall)=
## The Retrieval Texture Firewall

:::{div} feynman-prose
Now we come to a subtle but critical issue. When you retrieve external knowledge, it comes with all sorts of irrelevant details: the font it was written in, the exact phrasing, spelling variations, formatting artifacts. This is the "texture" of the document.

You want to *use* this texture when generating output---if the user asks for a quote, you should give them the exact quote. But you emphatically do *not* want this texture affecting your reasoning. If your policy starts depending on whether a source used British or American spelling, you've got a problem.

The solution is a firewall: a strict separation between what goes to the control loop and what goes to the decoder. The control loop sees only $(K, z_n)$---the semantic content. The decoder sees the full atom including $z_{\text{tex}}$. The firewall ensures that $\partial\pi / \partial z_{\text{tex}} = 0$---texture never influences policy.
:::

:::{prf:definition} Bulk Projection Operator
:label: def-bulk-projection-operator

The *bulk projection* $\Pi_{\text{bulk}}: \mathcal{Z}_{\text{ext}} \to \mathcal{K} \times \mathcal{Z}_n$ is defined by:

$$
\Pi_{\text{bulk}}(\xi) = \Pi_{\text{bulk}}(K, z_n, z_{\text{tex}}) := (K, z_n).

$$
*Interpretation:* This projection discards texture, retaining only control-relevant coordinates.

*Cross-reference:* This extends the internal texture exclusion of {ref}`sec-conditional-independence-and-sufficiency` to external retrieval.

:::
:::{prf:definition} Bulk-Filtered Retrieval Potential
:label: def-bulk-filtered-retrieval-potential

The *retrieval potential* is:

$$
\Psi_{\text{ret}}(z) = -\Lambda_{\text{ret}} \int_{\mathcal{Z}_{\text{ext}}} \exp\left(-\lambda \, d_{\mathbb{D}}(z, \Pi_{\text{bulk}}(\xi))\right) d\nu_\omega(\xi),

$$
with the firewall constraint:

$$
\frac{\partial \Psi_{\text{ret}}}{\partial z_{\text{tex,ext}}} \equiv 0.

$$
*Units:* $[\Psi_{\text{ret}}] = \text{nat}$, $[\Lambda_{\text{ret}}] = \text{nat}$, $[\lambda] = [z]^{-1}$.

*Cross-reference:* Compare the memory potential $\Psi_{\text{mem}}$ (Definition {prf:ref}`def-memory-potential`), which uses heat kernel rather than geodesic exponential. Both generate conservative forces.

:::
:::{prf:theorem} Stability of Retrieval Loop
:label: thm-stability-of-retrieval-loop

Under the firewall constraint (Definition {prf:ref}`def-bulk-filtered-retrieval-potential`), the retrieval force field:

$$
\mathbf{f}_{\text{ret}} = -G^{-1}\nabla_G \Psi_{\text{ret}}

$$
is smooth (Lipschitz in $z$) and independent of external texture coordinates $z_{\text{tex,ext}}$.

*Consequence:* The control loop remains stable; external texture cannot inject high-frequency gradients that would trigger Mode T.C (Labyrinthine Overfitting).

*Proof sketch:* The bulk projection $\Pi_{\text{bulk}}$ is a smooth submersion. Composition with the smooth geodesic exponential preserves smoothness. The firewall constraint ensures $\nabla_{z_{\text{tex,ext}}} \Psi_{\text{ret}} = 0$ by construction. $\square$

*Cross-reference:* This theorem extends TextureFirewallCheck (Node 29) to external retrieval. See {ref}`sec-failure-modes` for Mode T.C classification.

**Heuristic 28.3.4 (Side-Channel Texture Delivery).**
External texture $z_{\text{tex,ext}}$ is delivered to the decoder via a side channel:
1. At stopping radius $R_{\text{cutoff}}$ ({ref}`sec-the-retrieval-texture-firewall`), retrieve the full atom $\xi = (K, z_n, z_{\text{tex}})$
2. Inject $z_{\text{tex}}$ directly to decoder attention, bypassing the EoM
3. The control loop only sees $(K, z_n)$

*Interpretation:* This is the retrieval analog of "reading a document without letting its style affect your reasoning."

:::

:::{div} feynman-prose
The "side-channel texture delivery" heuristic is worth understanding deeply. When the agent retrieves a document, two things happen in parallel:

1. The semantic content $(K, z_n)$ enters the control loop, potentially changing what action the agent takes.
2. The full content including texture goes to a buffer that the decoder can access during generation.

This separation ensures that the agent's *reasoning* depends only on *what* the document says, not *how* it says it. But the agent's *output* can faithfully reproduce the specific wording when needed.

It's like the difference between understanding an argument and quoting it verbatim. Understanding should be style-independent; quoting should preserve style exactly.
:::

(sec-retrieval-augmented-equations-of-motion)=
## Retrieval-Augmented Equations of Motion

:::{div} feynman-prose
Now we extend the equations of motion one more time. We had local potentials, then we added memory, and now we add retrieval. The pattern is exactly the same: each source of influence contributes a potential, and the agent feels the gradient of the sum.

The full equation now has three potential terms: $\Phi_{\text{eff}}$ (local value), $\Psi_{\text{mem}}$ (self-interaction from past trajectory), and $\Psi_{\text{ret}}$ (interaction with external knowledge). The agent is pulled simultaneously by where value is high, where it succeeded before, and where relevant knowledge lives.
:::

:::{prf:definition} Retrieval-Augmented Geodesic SDE
:label: def-retrieval-augmented-geodesic-sde

The equations of motion with retrieval are:

$$
dz^k = \left[\mathcal{M}_{\text{curl}}\right]^k{}_{j}\left(-G^{j\ell}\partial_\ell(\Phi_{\text{eff}} + \Psi_{\text{mem}} + \Psi_{\text{ret}}) + u_\pi^j\right) ds - \Gamma^k_{ij}\dot{z}^i\dot{z}^j\,ds + \sqrt{2T_c}(G^{-1/2})^{kj}dW^j_s,

$$
where:
- $\Phi_{\text{eff}}$: effective potential (Definition {prf:ref}`def-effective-potential`)
- $\Psi_{\text{mem}}$: memory potential (Definition {prf:ref}`def-memory-potential`)
- $\Psi_{\text{ret}}$: retrieval potential (Definition {prf:ref}`def-bulk-filtered-retrieval-potential`)
- $\Gamma^k_{ij}$: Christoffel symbols (Definition 2.5.1, Definition 22.2.1a)
- $u_\pi^k$: policy control field (Definition {prf:ref}`def-the-control-field`)
- $T_c$: cognitive temperature ({ref}`sec-the-geodesic-baoab-integrator`)

*Cross-reference:* This extends the memory-augmented SDE (Definition {prf:ref}`def-memory-augmented-geodesic-sde`) with the retrieval term $\Psi_{\text{ret}}$.

:::
:::{prf:proposition} Superposition of Non-Local Forces
:label: prop-superposition-of-non-local-forces

The total non-local force is:

$$
\mathbf{f}_{\text{non-local}} = -G^{-1}\nabla_G(\Psi_{\text{mem}} + \Psi_{\text{ret}}),

$$
where:
- Memory force $\mathbf{f}_{\text{mem}}$ integrates over the agent's past trajectory
- Retrieval force $\mathbf{f}_{\text{ret}}$ integrates over the external archive

*Interpretation:* The agent simultaneously experiences attraction to its own memory ({ref}`sec-section-non-local-memory-as-self-interaction-functional`) and to relevant external knowledge (this section).

:::
(sec-wfr-dynamics-retrieval-induced-mass-injection)=
## WFR Dynamics: Retrieval-Induced Mass Injection

:::{div} feynman-prose
Just as memory creates a source term in the WFR continuity equation, so does retrieval. But there's an important difference in interpretation. Memory mass creation is about the agent's own experience. Retrieval mass creation is about importing external information.

When you retrieve a highly relevant fact, it's as if belief mass spontaneously appears at the corresponding location in latent space. You weren't there before, and you didn't walk there---you just suddenly consider it possible because you read something relevant.
:::

:::{prf:definition} Retrieval Source Term
:label: def-retrieval-source-term

The Wasserstein–Fisher–Rao continuity equation with retrieval is:

$$
\partial_s \rho + \nabla \cdot (\rho \mathbf{v}) = \rho \, r_{\text{local}}(z) + \sigma_{\text{ret}}(z),

$$
where:
- $r_{\text{local}}(z)$: local mass creation rate (reward-driven, Definition {prf:ref}`def-the-wfr-action`)
- $\sigma_{\text{ret}}(z)$: retrieval source term

The retrieval source is:

$$
\sigma_{\text{ret}}(z) = \eta_{\text{ret}} \cdot \Psi_{\text{ret}}(z) \cdot \mathbf{1}[\Psi_{\text{ret}}(z) > \Psi_{\text{threshold}}],

$$
with $[\sigma_{\text{ret}}] = \text{nat}/[z]^d/\text{step}$.

*Cross-reference:* Compare {ref}`sec-wfr-dynamics-with-memory-sources`'s memory mass creation. Both mechanisms inject mass at non-local locations.

:::
:::{prf:proposition} Non-Causal Transition via Retrieval
:label: prop-non-causal-transition-via-retrieval

Mass injection at retrieved locations enables transitions without continuous geodesic paths:

$$
\rho(z', s + \Delta s) > 0 \quad \text{even if} \quad d_G(z, z') > \sup_{0 \leq \tau \leq \Delta s} \|\mathbf{v}(z, s+\tau)\| \cdot \Delta s.

$$
*Interpretation:* Retrieval teleports probability mass to semantically relevant regions, bypassing the diffusion constraint. This is the WFR-level description of "jumping to a retrieved fact."

:::

:::{div} feynman-prose
"Non-causal transition" sounds paradoxical, but it makes perfect sense once you see it. Normally, probability can only spread gradually---you can't jump from A to B without passing through the space between. The diffusion process imposes a speed limit.

But retrieval breaks this. If you retrieve a fact about B while you're at A, probability suddenly appears at B. No traversal required. This is the mathematical formalization of "suddenly realizing something" or "making a connection you hadn't made before." The insight doesn't have to diffuse slowly through intermediate states; it can just appear.

This is immensely powerful for reasoning, but it also requires careful control. Uncontrolled teleportation would make the dynamics chaotic. The threshold in the retrieval source term ensures that only sufficiently relevant retrievals trigger mass injection.
:::

(sec-diagnostic-nodes-for-retrieval-integrity)=
## Diagnostic Nodes for Retrieval Integrity

:::{div} feynman-prose
Retrieval is powerful, but it can go wrong in ways that are hard to detect without explicit monitoring. We need two kinds of checks:

1. **Alignment check**: Is the external knowledge base actually living in the same geometric space as the agent's internal representations? If the embeddings have drifted apart, retrieval will return irrelevant results.

2. **Firewall check**: Is the texture actually staying out of the control loop? If texture is leaking into policy gradients, the agent will start making decisions based on surface features rather than semantics.

These are the retrieval analogs of the diagnostics we've seen throughout the Sieve. You can't just assume things are working; you have to measure.
:::

We introduce two diagnostic nodes for monitoring retrieval health.

(node-44)=
**Node 44: HyperbolicAlignmentCheck**

| **#**  | **Name**                     | **Component** | **Type**           | **Interpretation**                       | **Proxy**                                                                                                                               | **Cost**                    |
|--------|------------------------------|---------------|--------------------|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|-----------------------------|
| **44** | **HyperbolicAlignmentCheck** | Interface     | Metric Consistency | Are internal/external manifolds aligned? | $\Delta_{\text{align}} := \mathbb{E}[\lVert d_{\mathbb{D}}^{\text{int}}(z, z') - d_{\mathbb{D}}^{\text{ext}}(\Phi(z), \Phi(z'))\rVert]$ | $O(\lVert\nu_\omega\rVert)$ |

**Interpretation:** Tests whether the isometry axiom (Axiom {prf:ref}`ax-metric-isometry`) holds empirically. Large $\Delta_{\text{align}}$ indicates embedding drift or domain shift between internal representations and external knowledge base.

**Threshold:** $\Delta_{\text{align}} < 0.1 \cdot \bar{d}_{\mathbb{D}}$ (alignment error below 10% of mean geodesic distance).

(node-45)=
**Node 45: RetrievalFirewallCheck**

| **#**  | **Name**                   | **Component** | **Type**         | **Interpretation**                         | **Proxy**                                                                                                  | **Cost**            |
|--------|----------------------------|---------------|------------------|--------------------------------------------|------------------------------------------------------------------------------------------------------------|---------------------|
| **45** | **RetrievalFirewallCheck** | Policy        | Causal Isolation | Is external texture isolated from control? | $\Gamma_{\text{leak}} := \lVert\nabla_{z_{\text{int}}} (\partial \pi / \partial z_{\text{tex,ext}})\rVert$ | $O(d_{\text{tex}})$ |

**Interpretation:** Measures texture leakage into policy gradients. Should be near-zero under Theorem {prf:ref}`thm-stability-of-retrieval-loop`.

**Threshold:** $\Gamma_{\text{leak}} < \epsilon_{\text{firewall}}$ (implementation-dependent; typically $10^{-6}$).

*Cross-reference:* Node 45 extends the internal TextureFirewallCheck (Node 29) to external retrieval.

(sec-summary)=
## Summary

:::{div} feynman-prose
Let's put memory and retrieval side by side to see how they're the same and how they're different.

They're the same in structure: both are non-local potentials that integrate over some external source (past trajectory or knowledge archive) and contribute forces to the equations of motion. Both create mass sources in the WFR dynamics. Both need diagnostic monitoring to stay healthy.

They're different in source and interpretation. Memory is about *you*---your own past experiences reaching forward to guide your current decisions. Retrieval is about *others*---external knowledge that can inform you. Memory uses the heat kernel (smooth, diffusive influence). Retrieval uses geodesic exponential (sharp, proximity-based selection). Memory's firewall is temporal (past vs present). Retrieval's firewall is spatial (bulk vs texture).

Together, they give the agent two complementary ways to transcend the tyranny of local gradients.
:::

**Table 28.7.1 (Memory vs Retrieval Comparison).**

| Aspect         | Memory ({ref}`sec-section-non-local-memory-as-self-interaction-functional`)                                        | Retrieval ({ref}`sec-section-hyperbolic-active-retrieval-geodesic-search-and-semantic-pull-back`)                                                      |
|----------------|------------------------------------------------------------|-----------------------------------------------------------------------------|
| **Source**     | Internal trajectory $\gamma_{0:T}$                         | External archive $\mathcal{Z}_{\text{ext}}$                                 |
| **Kernel**     | Heat kernel $H_\tau(z, z')$                                | Geodesic exponential $\exp(-\lambda d_{\mathbb{D}})$                        |
| **Potential**  | $\Psi_{\text{mem}}$ (Def. {prf:ref}`def-memory-potential`) | $\Psi_{\text{ret}}$ (Def. {prf:ref}`def-bulk-filtered-retrieval-potential`) |
| **Firewall**   | Temporal (past vs present)                                 | Spatial (bulk vs texture)                                                   |
| **WFR source** | $r_{\text{mem}}(z)$                                        | $\sigma_{\text{ret}}(z)$                                                    |
| **Diagnostic** | Node 43 (MemoryBalanceCheck)                               | Nodes 44–45                                                                 |

*Key insight:* Memory and retrieval are dual non-local mechanisms. Memory integrates over temporal history; retrieval integrates over spatial archive. Both contribute conservative forces to the equations of motion (Definition {prf:ref}`def-retrieval-augmented-geodesic-sde`) and mass sources to WFR dynamics (Definition {prf:ref}`def-retrieval-source-term`).



(sec-bilevel-nonlocal-regulation)=
## Bilevel Regulation of Non-Local Potentials (Joint Optimization Resolution)

:::{div} feynman-prose
Now we face a practical question: how much should the agent rely on memory versus retrieval? This isn't something you can set once and forget. The right balance depends on the situation. In a familiar environment where your past experience is a good guide, lean on memory. In a novel situation where you need external information, lean on retrieval.

The Universal Governor (Section 26) handles this automatically. It watches the diagnostic signals---is the agent surprised by reality? is memory dominating too much?---and adjusts the strengths $\Lambda_{\text{mem}}$ and $\Lambda_{\text{ret}}$ accordingly.

The key insight is that "surprise" (the Interventional Gap) tells you whether your internal model is adequate. If you're not surprised, your memory is working well. If you're surprised, you need external information. The Governor translates this into concrete adjustments.
:::

The joint optimization of memory strength $\Lambda_{\text{mem}}$ and retrieval strength $\Lambda_{\text{ret}}$ is solved by the Universal Governor ({ref}`sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller`) acting on the diagnostic residuals of Nodes 43 and 53.

:::{prf:proposition} Optimal Non-Local Coupling
:label: prop-optimal-nonlocal-coupling

Let the control vector be $\Lambda = (\Lambda_{\text{mem}}, \Lambda_{\text{ret}})$. The optimal coupling is the fixed point of the Governor's policy $\pi_{\mathfrak{G}}$ ({prf:ref}`def-the-universal-governor`) given the diagnostic state $s_t = (\Delta_{\text{causal}}, \Omega_{\text{mem}})$.

**Control Law Derivation:**

1. **Surprise Signal:** Let $\Delta_{\text{causal}} = D_{\text{KL}}(P_{\text{int}} \| P_{\text{obs}})$ be the Interventional Gap (Node 53).

2. **Overfitting Signal:** Let $\Omega_{\text{mem}}$ be the Non-Locality Ratio ({prf:ref}`def-non-locality-ratio`, Node 43).

3. **Governor Update:** The Lyapunov descent condition $\Delta V_{\mathfrak{L}} < 0$ ({prf:ref}`def-training-lyapunov-function`) implies the following qualitative update dynamics:

$$
\begin{aligned}
\dot{\Lambda}_{\text{ret}} &\propto \alpha_1 \cdot \Delta_{\text{causal}} \\
\dot{\Lambda}_{\text{mem}} &\propto \alpha_2 \cdot (\Delta_{\text{causal}}^{\text{target}} - \Delta_{\text{causal}}) - \alpha_3 \cdot \operatorname{ReLU}(\Omega_{\text{mem}} - \Omega_{\max})
\end{aligned}

$$
where $\alpha_1, \alpha_2, \alpha_3 > 0$ are learning rates and $\Omega_{\max}$ is the maximum tolerable non-locality ratio.

*Proof sketch.* The Governor's outer objective ({prf:ref}`def-outer-problem-governor-optimization`) includes terms penalizing both prediction error (Interventional Gap) and overfitting (Non-Locality Ratio). The gradient of this objective with respect to $\Lambda$ yields the stated control law. At equilibrium, $\dot{\Lambda} = 0$, which implies a balance between reliance on memory and retrieval calibrated to the agent's surprise level. $\square$
:::

:::{prf:remark} Operational Interpretation
:label: rem-memory-retrieval-interpretation

- **If the agent is surprised by reality** ($\Delta_{\text{causal}}$ high): It must increase reliance on external truth ($\Lambda_{\text{ret}} \uparrow$).
- **If the agent is not surprised** ($\Delta_{\text{causal}}$ low): It can conserve bandwidth by relying on internal memory ($\Lambda_{\text{mem}} \uparrow$), subject to the constraint that it must not overfit ($\Omega_{\text{mem}} < \Omega_{\max}$).

This closes the joint optimization problem by reducing it to a specific instantiation of the Governor's Lyapunov stability framework ({prf:ref}`def-training-lyapunov-function`).
:::



(sec-safe-retrieval-bandwidth)=
## The Safe Retrieval Bandwidth Corollary (Instability Resolution)

:::{div} feynman-prose
There's a limit to how much information you can stuff into the agent's latent space. This is the holographic bound from Section 33---the total information content must not exceed the interface capacity.

Retrieval adds information. If you retrieve too much, you can push the system past its capacity limit. When that happens, the metric becomes singular and the dynamics freeze up. It's like trying to pour a gallon of water into a pint glass---the excess has nowhere to go, and the system breaks.

This theorem tells you exactly when that happens and what to do about it: either increase your interface bandwidth (more sensors, bigger observation space) or reduce retrieval intensity. You can't have infinite knowledge in a finite system.
:::

Retrieval-induced instability is identified as the violation of the **Causal Information Bound** ({ref}`sec-causal-information-bound`). Retrieval functions as a mass-injection source term; stability is preserved only if the total bulk information respects the interface area law.

:::{prf:theorem} Safe Retrieval Bandwidth
:label: thm-safe-retrieval-bandwidth

Let $\sigma_{\text{ret}}(z)$ be the retrieval source term in the WFR continuity equation ({prf:ref}`def-retrieval-source-term`). The latent geometry remains non-singular if and only if the total information flux satisfies:

$$
\int_{\mathcal{Z}} \left( \rho_I(z) + \sigma_{\text{ret}}(z) \right) \, d\mu_G \leq \kappa \, C_{\partial}

$$
where $C_{\partial} = \nu_D \cdot \text{Area}(\partial\mathcal{Z})/\ell_L^{D-1}$ is the boundary capacity (Definition {prf:ref}`def-holographic-coefficient`, {prf:ref}`def-levin-length`).

*Proof.*
1. **Mass Augmentation:** Retrieval modifies the bulk information density: $\tilde{\rho}_I = \rho_I + \sigma_{\text{ret}}$.

2. **Metric Response:** By the Capacity-Constrained Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`), the radial metric component scales as $G_{rr} \propto (1 - \tilde{I}_{\text{bulk}}/C_{\partial})^{-1}$.

3. **Singularity:** If $\int \sigma_{\text{ret}} > C_{\partial} - I_{\text{bulk}}$, then $G_{rr} \to \infty$ at a radius $r < 1$ (the horizon moves inward).

4. **Dynamical Consequence:** The update velocity $\|v\|_G \to 0$ (Causal Stasis, {ref}`sec-causal-information-bound`). The instability manifests as the freezing of the agent's inference dynamics due to saturation of the holographic bound. $\square$
:::

*Interpretation:* External retrieval becomes destabilizing when it pushes the total information content beyond the holographic capacity of the interface. The remedy is to increase interface bandwidth (more sensors) or reduce retrieval intensity.



(sec-causal-isometry-theorem)=
## The Causal Isometry Theorem (Cross-Modal Retrieval Resolution)

:::{div} feynman-prose
Here's a beautiful result that justifies cross-modal retrieval---using information from images to inform language processing, or vice versa.

The claim is this: if two different sensory channels (say, vision and language) both provide enough information to solve the same control task, then their latent geometries *must* be isometric---they must be the same shape, up to relabeling. Why? Because the geometry is determined by the causal structure of the task, not by the sensory channel. The task is invariant; the geometry follows the task.

This is why you can train image embeddings and text embeddings separately and then use them together: if they're solving the same semantic tasks, they must be organizing concepts the same way. The isometry isn't a happy accident; it's a consequence of the physics.
:::

We prove that if two modalities allow for the solution of the same causal control task, their capacity-constrained geometries must be isometric in the bulk.

:::{prf:theorem} Causal Isometry Theorem
:label: thm-causal-isometry

Let $\mathcal{M}_A$ and $\mathcal{M}_B$ be latent manifolds encoding modalities $A$ and $B$ of a common environment $\mathcal{E}$. Let $\Phi_{\text{causal}}$ be the Causal Information Potential ({ref}`sec-causal-discovery-interventional-geometry-and-the-singularity-of-action`). If both representations are **Interventionally Closed** ({prf:ref}`thm-interventional-closure`), then the induced metrics $G_A$ and $G_B$ are isometric.

*Proof.*
1. **Metric Genesis:** According to the Capacity-Constrained Metric Law ({prf:ref}`thm-capacity-constrained-metric-law`), the metric $G$ is determined by the solution to the Einstein-like equation $R_{ij} - \frac{1}{2}R G_{ij} + \Lambda G_{ij} = \kappa T_{ij}$, where the stress-energy tensor $T_{ij}$ is derived from the risk Lagrangian $\mathcal{L}_{\text{risk}}$.

2. **Risk Invariance:** The risk Lagrangian $\mathcal{L}_{\text{risk}}(V) = \frac{1}{2}\|\nabla_A V\|^2 + U(V)$ depends only on the Value function $V$ and the Causal Potential $\Psi_{\text{causal}}$.

3. **Task Invariance:** The potentials $V$ and $\Psi_{\text{causal}}$ are functions of the *causal graph* of the environment $\mathcal{E}$, which is an invariant independent of the sensory modality (pixels vs. tokens).

4. **Uniqueness:** Assuming the solution to the metric field equation is unique (guaranteed for the Poincare disk ansatz in the saturation limit), the geometries $G_A$ and $G_B$ are identical up to a diffeomorphism determined by the encoder parameterization. $\square$
:::

*Interpretation:* Latent representations of the same concept in different modalities (e.g., visual vs. textual) are geometrically isometric because the risk functional governing the metric depends only on the causal structure of the environment, not the sensory channel. This justifies cross-modal retrieval: information retrieved from one modality can inform reasoning in another if both are grounded in the same causal graph.
