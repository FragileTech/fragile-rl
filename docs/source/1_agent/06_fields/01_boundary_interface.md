(sec-the-boundary-interface-symplectic-structure)=
# The Boundary Interface: Symplectic Structure

## TLDR

- The agent–world interface is not just I/O plumbing: it has a **symplectic structure** where observations and actions
  are conjugate boundary conditions.
- Treat sensors/actions as boundary constraints on latent dynamics (Dirichlet/Neumann-style), making coupling auditable.
- This chapter defines the “contact surface” that later field equations (reward/value forms, information bounds) depend
  on.
- Operationally: it tells you what information must be exposed at the boundary and how to test coupling/grounding.
- The main failure modes are decoupling (ungrounded inference) and instability (feedback amplification); both are
  monitorable.

## Roadmap

1. Symplectic interface picture and why it is the right mathematical type.
2. How observations/actions appear as boundary conditions on internal dynamics.
3. Diagnostics for coupling and grounding at the interface.

{cite}`arnold1989mathematical`

:::{div} feynman-prose
All right, now we come to something that I find absolutely fascinating---the place where the agent meets the world. You see, so far we've been talking about what happens *inside* the agent, all this beautiful geometry and dynamics on the latent manifold. But an agent that doesn't touch reality isn't much of an agent, is it?

Here's the profound question: How does information get *in* and *out*? Not just "sensors provide data and motors send commands"---that's the boring answer. The interesting answer is that the interface between agent and world has a deep mathematical structure. It's a *symplectic manifold*, where observations and actions live as conjugate variables, just like position and momentum in physics.

This isn't a metaphor. It's the same mathematics. And understanding this connection will tell us exactly how sensors and motors work at the deepest level.
:::

(rb-boundary-conditions)=
:::{admonition} Researcher Bridge: Observations and Actions as Boundary Conditions
:class: info
In standard RL, observations and actions are inputs and outputs. Here they are boundary conditions on the latent dynamics, which is why sensor and motor channels appear as Dirichlet and Neumann conditions.
:::

:::{div} feynman-prose
We have defined the internal dynamics of the agent (the interior) as a Jump-Diffusion process on a Riemannian fiber bundle ({ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`). We now rigorously define its coupling to the external world.

The interface exchanges information with the environment via two asymmetric boundary conditions: **Dirichlet** (position-clamping for sensors) and **Neumann** (flux-clamping for motors).

Now, what do these terms mean? If you've taken a course in partial differential equations, you know that to solve an equation on a region, you need to specify what happens at the boundary. There are two classic choices:

- **Dirichlet**: You fix the *value* of the solution at the boundary. "The temperature at the wall is 100 degrees."
- **Neumann**: You fix the *flux* (the rate of flow) at the boundary. "Heat flows out through the wall at 50 watts per square meter."

The remarkable thing is that sensors and motors naturally correspond to these two cases. Sensors tell you *where* you are (position)---that's Dirichlet. Motors tell you *how fast* you're pushing (flux)---that's Neumann. This isn't a coincidence; it reflects a deep duality in physics and information theory.
:::

(sec-the-symplectic-interface-position-momentum-duality)=
## The Symplectic Interface: Position-Momentum Duality

:::{div} feynman-prose
Now we get to the heart of the matter. The boundary between agent and environment isn't just a wall or a membrane---it has *structure*. Specifically, it's a symplectic manifold.

What's a symplectic manifold? Let me give you the picture first, then the mathematics. Imagine a dance floor. At each point on the floor, you could be standing still, or moving in some direction, or spinning. Now, the symplectic structure is like a rule that says: if you know your position *and* your momentum, you know everything there is to know about your motion. Position and momentum together form a complete description.

But here's the beautiful part: they're not independent. They're *conjugate*. If you change your position, it affects how your momentum evolves, and vice versa. In classical mechanics, Hamilton's equations tell you exactly how: $\dot{q} = \partial H/\partial p$ and $\dot{p} = -\partial H/\partial q$. Position comes from momentum; momentum comes from position. They're locked in an eternal dance.

At the agent's interface, observations play the role of position (they tell you *where* you are in representation space), and actions play the role of momentum (they tell you *how* you're pushing). The symplectic structure captures this duality exactly.
:::

The boundary $\partial\mathcal{Z}$ between agent and environment is not merely a surface---it is a **symplectic manifold** where observations and actions live as conjugate variables.

:::{prf:definition} Symplectic Boundary Manifold
:label: def-symplectic-boundary-manifold

The agent's interface is a symplectic manifold $(\partial\mathcal{Z}, \omega)$ with canonical coordinates $(q, p) \in T^*\mathcal{M}$ where:
- $q \in \mathcal{Q}$ is the **position bundle** (sensory configuration)
- $p \in T^*_q\mathcal{Q}$ is the **momentum bundle** (motor flux)

The symplectic form is:

$$
\omega = \sum_{i=1}^n dq^i \wedge dp_i.

$$
Units: $[\omega] = [q][p] = \mathrm{nat}$.

*Remark (Causal Structure).* The symplectic structure encodes causality: observations fix "where" the belief state is (position), while actions fix "how" it flows outward (momentum/flux). These cannot be treated symmetrically as static fields.

:::

:::{div} feynman-prose
Let me unpack that definition a bit. The symplectic form $\omega = \sum_i dq^i \wedge dp_i$ might look like abstract nonsense, but it's actually telling you something very concrete. It says: when you integrate $\omega$ over any little patch of the phase space, you get the "area" of that patch in a very specific sense---the sense that's preserved by Hamiltonian dynamics. No matter how the system evolves, this area is conserved.

The units being "nat" (natural units of information) is crucial. This tells you that position-momentum pairs at the boundary carry information, and the symplectic structure measures how much. When you observe something (fix $q$), you're committing to a position in belief space. When you act (fix $p$), you're committing to a direction of push. Together, they determine a point in phase space, and that point carries information content measured by $\omega$.

The remark about causal structure is subtle but important: observations fix *where* you are; actions fix *how* you're moving. You can't swap them. This asymmetry is built into the mathematics through the distinction between Dirichlet and Neumann boundary conditions.
:::

:::{prf:definition} Dirichlet Boundary Condition --- Sensors
:label: def-dirichlet-boundary-condition-sensors

The sensory input stream $\phi(x)$ imposes a **Dirichlet** (position-clamping) condition on the belief density:

$$
\rho_{\partial}^{\text{sense}}(q, t) = \delta(q - q_{\text{obs}}(t)),

$$
where $q_{\text{obs}}(t) = E_\phi(x_t)$ is the encoded observation. This clamps the *configuration* of the belief state.

*Interpretation:* Information flow from environment to agent (observation).

:::

:::{div} feynman-prose
Think about what that delta function means. When you receive an observation, it *slams* your belief state to a specific location. Before the observation, you might have been uncertain about where you are in belief space---spread out, diffuse. After the observation, bang! You're at $q_{\text{obs}}$. No ambiguity. That's why it's called "clamping."

This is exactly what sensors do: they *localize* you. A camera tells you "the visual world looks like this." An accelerometer tells you "you're tilted by this much." Each observation pins down some aspect of your position in representation space.
:::

:::{prf:definition} Neumann Boundary Condition --- Motors
:label: def-neumann-boundary-condition-motors

The motor output stream $A(x)$ imposes a **Neumann** (flux-clamping) condition:

$$
\nabla \rho \cdot \mathbf{n} \big|_{\partial\mathcal{Z}_{\text{motor}}} = j_{\text{motor}}(p, t),

$$
where $j_{\text{motor}}$ is the motor current density determined by the policy:

$$
j_{\text{motor}} = D_A(u_\pi) = \text{Decoder}(z, u_\pi, z_{\text{tex,motor}}).

$$
*Interpretation:* Information flow from agent to environment (action).

Units: $[j_{\text{motor}}] = \mathrm{nat}/\text{step}$.

:::

:::{div} feynman-prose
Motors work differently from sensors. Instead of pinning down *where* you are, they specify *how much flow* is crossing the boundary. Think of it like this: a sensor is a window you look through; a motor is a faucet you turn on.

The motor doesn't care exactly where the belief state is. It cares about how much *influence* is flowing outward. The policy says "push this hard in that direction," and the motor boundary condition enforces that flux. The actual position can wiggle around, but the rate of flow is clamped.

This is why the mathematics uses the gradient (the $\nabla_n \rho \cdot \mathbf{n}$ term). The gradient tells you the rate of change---how much "stuff" is flowing across the boundary. For motors, we fix the flux, not the value.
:::

(pi-hamiltonian-bc)=
::::{admonition} Physics Isomorphism: Hamiltonian Boundary Conditions
:class: note

**In Physics:** In Hamiltonian mechanics, canonical coordinates $(q, p)$ satisfy $\dot{q} = \partial H/\partial p$ (position from momentum) and $\dot{p} = -\partial H/\partial q$ (momentum from position). Boundary conditions fix either $q$ (Dirichlet) or $\partial_n q \propto p$ (Neumann) {cite}`arnold1989mathematical`.

**In Implementation:** The agent's interface imposes dual boundary conditions:
- **Perception (Dirichlet):** Observations fix position $z|_{\partial\mathcal{Z}} = z_{\text{obs}}$
- **Action (Neumann):** Motors fix momentum flux $\partial_n z|_{\partial\mathcal{Z}} = u_\pi$
- **Reward (Source):** Boundary reward flux $J_r|_{\partial\mathcal{Z}}$ (scalar charge density
  $\sigma_r$ in the conservative case)

**Correspondence Table:**

| Hamiltonian Mechanics | Agent (Symplectic Interface) |
|:----------------------|:-----------------------------|
| Position $q$ | Latent state $z$ |
| Momentum $p$ | Policy gradient $\nabla_z V$ |
| Dirichlet BC $q\vert_{\Gamma} = q_0$ | Observation constraint |
| Neumann BC $\partial_n q\vert_{\Gamma} = f$ | Action constraint |
| Hamiltonian $H(q,p)$ | Effective potential $\Phi_{\text{eff}}$ ({prf:ref}`def-effective-potential`) |
::::

:::{prf:proposition} Symplectic Duality Principle
:label: prop-symplectic-duality-principle

Under the canonical transformation $(q, p) \mapsto (p, -q)$:
- Dirichlet conditions become Neumann conditions
- Sensors become motors
- Perception becomes action

This duality is the mathematical foundation for the symmetric treatment of sensing and actuation.

*Proof sketch.* The symplectic form $\omega$ is invariant under canonical transformations. The Legendre transform $\mathcal{L}: T\mathcal{Q} \to T^*\mathcal{Q}$ maps velocity to momentum, exchanging position-fixing (Dirichlet) for flux-fixing (Neumann). $\square$

**Cross-references:** {ref}`sec-the-interface-and-observation-inflow` (Observation inflow), Definition {prf:ref}`def-dirichlet-boundary-condition-sensors`.

:::

:::{div} feynman-prose
Now here's something that might blow your mind a little. The proposition says that if you do a canonical transformation---swap position and momentum---then Dirichlet becomes Neumann, sensors become motors, and perception becomes action.

This is the deep reason why sensing and acting have the same mathematical structure. They're *dual* to each other. It's like how in electricity, you can swap electric and magnetic fields (under certain conditions), and the equations still work. Here, you can swap observations and actions, and the boundary conditions still make sense.

This isn't just mathematical elegance. It has practical consequences: you can design perception systems and motor systems using the same principles, because they're two faces of the same coin. The Visual Atlas and Action Atlas that we'll define next exploit this duality.
:::

(sec-the-dual-atlas-architecture)=
## The Dual Atlas Architecture

:::{div} feynman-prose
Now that we understand the symplectic structure, we need to actually build something. How do you implement an interface that respects all this beautiful mathematics?

The answer is to use *atlases*---collections of charts that together cover the whole space. If you've studied differential geometry, you know that a manifold is defined by its atlas: a set of overlapping patches, each with its own coordinate system, with smooth transitions between them.

Here's the key insight: perception and action each need their own atlas, but these atlases are related by the Legendre transform. The Visual Atlas tells you "given what I see, where am I?" The Action Atlas tells you "given what I want to do, how do I push?" And the Legendre transform is the mathematical operation that connects them---the same operation that connects Lagrangian mechanics (position and velocity) to Hamiltonian mechanics (position and momentum).

Why do we need separate atlases? Because the same physical situation might look very different from the perception side versus the action side. When you're looking at a cup, the visual representation involves shape, color, distance. When you're reaching for that cup, the motor representation involves joint angles, velocities, forces. These are different coordinate systems on the same underlying reality, and the Legendre transform is what translates between them.
:::

To implement the symplectic interface, we require two symmetric topological structures: a **Visual Atlas** for perception and an **Action Atlas** for actuation. This symmetrizes the architecture from {ref}`sec-the-shutter-as-a-vq-vae`.

:::{prf:definition} Visual Atlas — Perception
:label: def-visual-atlas-perception

The Visual Atlas $\mathcal{A}_{\text{vis}} = \{(U_\alpha, \phi_\alpha, e_\alpha^{\text{vis}})\}_{\alpha \in \mathcal{K}_{\text{vis}}}$ is a chart atlas on the sensory manifold $\mathcal{Q}$ with:
- **Charts** $U_\alpha \subset \mathcal{Q}$: Objects, Scenes, Viewpoints
- **Chart maps** $\phi_\alpha: U_\alpha \to \mathbb{R}^{d_{\text{vis}}}$: Local coordinates
- **Codebook embeddings** $e_\alpha^{\text{vis}} \in \mathbb{R}^{d_m}$: Discrete macro codes

*Input:* Raw observations $\phi_{\text{raw}}$ (pixels, sensors).
*Output:* Latent state $z \in \mathcal{Z}$ (configuration).

:::

:::{div} feynman-prose
Notice what the Visual Atlas does. It takes the raw visual chaos---pixels, shapes, colors---and organizes it into a structured representation. The charts ($U_\alpha$) are like different "ways of seeing": one chart might specialize in recognizing faces, another in outdoor scenes, another in small objects. The codebook embeddings ($e_\alpha^{\text{vis}}$) are the discrete labels: "this is a face," "this is a tree."

The output is a position in the latent space $\mathcal{Z}$. Every time you see something, the Visual Atlas tells you where you've landed in this internal coordinate system.
:::

:::{prf:definition} Action Atlas --- Actuation
:label: def-action-atlas-actuation

The Action Atlas $\mathcal{A}_{\text{act}} = \{(V_\beta, \psi_\beta, e_\beta^{\text{act}})\}_{\beta \in \mathcal{K}_{\text{act}}}$ is a chart atlas on the motor manifold $T^*\mathcal{Q}$ with:
- **Charts** $V_\beta \subset T^*\mathcal{Q}$: Gaits, Grasps, Tool Affordances (topologically distinct control regimes)
- **Chart maps** $\psi_\beta: V_\beta \to \mathbb{R}^{d_{\text{act}}}$: Local motor coordinates
- **Codebook embeddings** $e_\beta^{\text{act}} \in \mathbb{R}^{d_m}$: Action primitive codes

*Input:* Intention $u_{\text{intent}} \in T_z\mathcal{Z}$ (from Policy, {ref}`sec-policy-control-field`).
*Output:* Actuation $a_{\text{raw}}$ (torques, voltages).

*Remark (Jump Operator in Action Atlas).* The **Jump Operator** $L_{\beta \to \beta'}$ in the Action Atlas represents **Task Switching**: transitioning from one control primitive to another (e.g., "Walk" $\to$ "Jump", "Grasp" $\to$ "Release"). This mirrors the chart transition operator in the Visual Atlas ({ref}`sec-the-unified-world-model`).

:::

:::{div} feynman-prose
The Action Atlas mirrors the Visual Atlas, but on the motor side. Instead of "ways of seeing," you have "ways of doing." One chart might be for walking, another for grasping, another for using a tool. Each represents a topologically distinct control regime---you can't smoothly interpolate from walking to grasping; you have to *switch* between them.

The Jump Operator is how you switch. It's the motor equivalent of a saccade in vision: a discrete transition from one mode of operation to another. When you stop walking and start reaching for something, you've jumped between charts in the Action Atlas.

And here's the beautiful thing: the Legendre transform connects these two atlases. It's not that we designed them to be similar---they *have to be* similar, because they're related by a canonical transformation. This is why well-designed robot systems often have the same architecture for perception and control, just applied to different modalities.
:::

:::{prf:theorem} Atlas Duality via Legendre Transform
:label: thm-atlas-duality-via-legendre-transform

The Visual and Action Atlases are related by the Legendre transform $\mathcal{L}: T\mathcal{Q} \to T^*\mathcal{Q}$:

$$
\mathcal{A}_{\text{act}} = \mathcal{L}(\mathcal{A}_{\text{vis}}),

$$
where the chart transition functions satisfy:

$$
\psi_\beta \circ \mathcal{L} \circ \phi_\alpha^{-1} = \nabla_{\dot{q}} L(q, \dot{q})

$$
for Lagrangian $L(q, \dot{q}) = \frac{1}{2}\|\dot{q}\|_G^2 - V(q)$.

*Proof.* **Step 1 (Legendre transform definition).** The Legendre transform of a convex Lagrangian $L(q,\dot{q})$ is defined by:

$$
\mathcal{L}: T\mathcal{Q} \to T^*\mathcal{Q}, \qquad (q, \dot{q}) \mapsto \left(q, \frac{\partial L}{\partial \dot{q}}\right).

$$
For $L = \frac{1}{2}\|\dot{q}\|_G^2 - V(q)$, this gives $p = G(q)\dot{q}$, which is invertible when $G > 0$.

**Step 2 (Symplectic preservation).** The Legendre transform is a diffeomorphism that pulls back the canonical symplectic form $\omega_{T^*\mathcal{Q}} = dp \wedge dq$ to the Poincare-Cartan form $\omega_{T\mathcal{Q}} = d\theta_L$ where $\theta_L = \frac{\partial L}{\partial \dot{q}^i}dq^i$. This ensures that Hamiltonian flow on $T^*\mathcal{Q}$ corresponds to Lagrangian flow on $T\mathcal{Q}$.

**Step 3 (Chart compatibility).** Let $(U_\alpha, \phi_\alpha)$ be a chart in $\mathcal{A}_{\text{vis}}$ with coordinates $(q^\alpha, \dot{q}^\alpha)$. Define the induced action chart $(V_\beta, \psi_\beta)$ by $V_\beta = \mathcal{L}(U_\alpha \times T_{U_\alpha}\mathcal{Q})$ with coordinates $(q^\alpha, p^\alpha)$. The transition function is:

$$
\psi_\beta \circ \mathcal{L} \circ \phi_\alpha^{-1}: (q^\alpha, \dot{q}^\alpha) \mapsto (q^\alpha, G_{\alpha\beta}(q)\dot{q}^\beta),

$$
which is smooth and invertible by positive-definiteness of $G$. $\square$

*Remark (Why Legendre?).* The Legendre transform is the unique smooth map relating configuration-velocity (perception) to configuration-momentum (action) that:
1. Preserves the symplectic structure (Proposition {prf:ref}`prop-symplectic-duality-principle`)
2. Interchanges Dirichlet and Neumann boundary conditions ({ref}`sec-the-symplectic-interface-position-momentum-duality`)
3. Maps kinetic energy to Hamiltonian dynamics

*Cross-reference:* The metric $G$ appearing here is the capacity-constrained metric from Theorem {prf:ref}`thm-capacity-constrained-metric-law`, ensuring that the "mass" in the Legendre relation $p = G\dot{q}$ is the same "mass" that determines geodesic inertia (Definition {prf:ref}`def-mass-tensor`).

:::

(pi-legendre-transform)=
::::{admonition} Physics Isomorphism: Legendre Transform
:class: note

**In Physics:** The Legendre transform maps between Lagrangian and Hamiltonian formulations: $H(q,p) = p\dot{q} - L(q,\dot{q})$ where $p = \partial L/\partial \dot{q}$. It exchanges velocity for momentum as the independent variable {cite}`arnold1989mathematical`.

**In Implementation:** The Visual and Action Atlases are related by Legendre duality (Theorem {prf:ref}`thm-atlas-duality-via-legendre-transform`):

$$
\mathcal{L}: T\mathcal{Q} \to T^*\mathcal{Q}, \quad (z, \dot{z}) \mapsto (z, p = G\dot{z})

$$
**Correspondence Table:**
| Analytical Mechanics | Agent (Symplectic Interface) |
|:---------------------|:-----------------------------|
| Configuration space $\mathcal{Q}$ | Latent state space $\mathcal{Z}$ |
| Tangent bundle $T\mathcal{Q}$ | Velocity representation |
| Cotangent bundle $T^*\mathcal{Q}$ | Momentum representation |
| Lagrangian $L(q,\dot{q})$ | Kinetic action |
| Hamiltonian $H(q,p)$ | Effective potential $\Phi_{\text{eff}}$ |
| Velocity $\dot{q}$ | Policy output $u_\pi$ |
| Momentum $p$ | Value gradient $\nabla_A V$ |

**Duality:** Perception (Dirichlet) fixes position; Action (Neumann) fixes momentum flux.
::::

:::{prf:definition} The Holographic Shutter — Unified Interface
:label: def-the-holographic-shutter-unified-interface

The Shutter is extended from {ref}`sec-the-shutter-as-a-vq-vae` to a symmetric tuple:

$$
\mathbb{S} = (\mathcal{A}_{\text{vis}}, \mathcal{A}_{\text{act}}),

$$
where:
- **Ingress (Perception):** $E_\phi: \mathcal{Q} \to \mathcal{Z}$ via Visual Atlas
- **Egress (Actuation):** $D_A: T_z\mathcal{Z} \times \mathcal{Z} \to T^*\mathcal{Q}$ via Action Atlas
- **Proprioception (Inverse Model):** $E_A: T^*\mathcal{Q} \to T_z\mathcal{Z}$ maps realized actions back to intentions

**Cross-references:** {ref}`sec-the-shutter-as-a-vq-vae` (VQ-VAE Shutter), {ref}`sec-tier-the-attentive-atlas` (AttentiveAtlasEncoder), {ref}`sec-decoder-architecture-overview-topological-decoder` (TopologicalDecoder).

:::
(sec-motor-texture-the-action-residual)=
## Motor Texture: The Action Residual

:::{div} feynman-prose
Now we come to something subtle but important. When you reach for a cup, your brain doesn't specify the exact position of every muscle fiber at every millisecond. It specifies something more abstract: "reach toward that location with this general trajectory." The fine details---the slight tremor in your fingers, the micro-adjustments for balance, the precise timing of individual motor units---those emerge from lower-level systems.

This is *motor texture*. It's the high-frequency, fine-grained detail of motor execution that doesn't matter for planning. Just like visual texture (the exact pixel values in an image) doesn't matter for recognizing what object you're looking at, motor texture doesn't matter for deciding what action to take.

The reason this matters is the sim-to-real gap. In simulation, your motors are perfect: no tremor, no noise, no friction. In reality, all of that exists. If your policy depends on motor texture, it will fail catastrophically in the real world. So we build a *firewall*: the policy never sees motor texture, and therefore can't depend on it. The texture is only used for low-level execution, not for decision-making.
:::

Just as visual texture captures reconstruction-only detail ({ref}`sec-the-retrieval-texture-firewall`), **motor texture** captures actuation-only detail that is excluded from planning.

:::{prf:definition} Motor Texture Decomposition
:label: def-motor-texture-decomposition

The motor output decomposes as:

$$
a_t = (K^{\text{act}}_t, z_{n,\text{motor}}, z_{\text{tex,motor}}),

$$
where:
- $K^{\text{act}}_t \in \mathcal{K}_{\text{act}}$ is the **discrete motor macro** (action primitive/chart index)
- $z_{n,\text{motor}} \in \mathbb{R}^{d_{\text{motor},n}}$ is **motor nuisance** (impedance, compliance, force distribution)
- $z_{\text{tex,motor}} \in \mathbb{R}^{d_{\text{motor,tex}}}$ is **motor texture** (tremor, fine-grained noise, micro-corrections)

*Remark (Parallel to Visual Decomposition).* This mirrors the visual decomposition $(K_t, z_{n,t}, z_{\text{tex},t})$ from {ref}`sec-the-shutter-as-a-vq-vae`:

| Component                 | Visual Domain                 | Motor Domain                              |
|---------------------------|-------------------------------|-------------------------------------------|
| **Macro (discrete)**      | Object/Scene chart $K$        | Action primitive $K^{\text{act}}$         |
| **Nuisance (continuous)** | Pose/viewpoint $z_n$          | Compliance/impedance $z_{n,\text{motor}}$ |
| **Texture (residual)**    | Pixel detail $z_{\text{tex}}$ | Tremor/noise $z_{\text{tex,motor}}$       |

:::
:::{prf:definition} Compliance Tensor
:label: def-compliance-tensor

The motor nuisance encodes the **compliance tensor**:

$$
C_{ij}(z_{n,\text{motor}}) = \frac{\partial a^i}{\partial f^j},

$$
where $f$ is the external force/feedback. This determines how the motor output responds to perturbations:
- **High compliance** ($C$ large): Soft, yielding response (safe interaction)
- **Low compliance** ($C$ small): Stiff, precise response (accurate positioning)

Units: $[C_{ij}] = [a]/[f]$.

:::
:::{prf:definition} Motor Texture Distribution
:label: def-motor-texture-distribution

At the motor boundary, texture is sampled from a geometry-dependent Gaussian:

$$
z_{\text{tex,motor}} \sim \mathcal{N}(0, \Sigma_{\text{motor}}(z)),

$$
where:

$$
\Sigma_{\text{motor}}(z) = \sigma_{\text{motor}}^2 \cdot G_{\text{motor}}^{-1}(z) = \sigma_{\text{motor}}^2 \cdot \frac{(1-|z|^2)^2}{4} I_{d_{\text{motor,tex}}}.

$$
This follows the same conformal scaling as visual texture (Definition {prf:ref}`def-boundary-texture-distribution`), ensuring consistent thermodynamic behavior.

:::
:::{prf:axiom} Motor Texture Firewall
:label: ax-motor-texture-firewall

Motor texture is decoupled from the Bulk dynamics:

$$
\partial_{z_{\text{tex,motor}}} \dot{z} = 0, \qquad \partial_{z_{\text{tex,motor}}} u_\pi = 0.

$$
The policy $\pi_\theta$ operates on $(K, z_n, A, z_{n,\text{motor}})$ but **never** on $(z_{\text{tex}}, z_{\text{tex,motor}})$.

*Remark (Sim-to-Real Gap).* The **motor texture variance** $\sigma_{\text{motor}}^2$ is the mathematical definition of the "Sim-to-Real gap":
- **Simulation:** $\sigma_{\text{motor}} \approx 0$ (deterministic, no tremor)
- **Reality:** $\sigma_{\text{motor}} > 0$ (friction, sensor noise, motor tremor)
- **Robustness:** The Bulk policy $u_\pi$ is invariant; only the Action Decoder learns to manage domain-specific noise.

**Cross-references:** {ref}`sec-the-retrieval-texture-firewall` (Texture Firewall), Axiom {prf:ref}`ax-bulk-boundary-decoupling`.

:::
(sec-the-belief-evolution-cycle-perception-dreaming-action)=
## The Belief Evolution Cycle: Perception--Dreaming--Action

:::{div} feynman-prose
All right, now we're going to tie everything together with one of the most beautiful pictures in this whole framework: the thermodynamic cycle of cognition.

Think about a heat engine. It compresses gas, heats it, expands it, cools it, and repeats. At each stage, energy and entropy flow in predictable ways. A Carnot engine achieves maximum efficiency by carefully managing these flows.

The agent does something remarkably similar. It has three phases:

1. **Perception (Compression)**: You receive a high-entropy sensory stream---millions of pixels, thousands of sensor readings---and compress it down to a low-entropy internal state. This is like compressing gas: you're squeezing information into a smaller representation. Entropy decreases inside the agent.

2. **Dreaming (Isentropic Evolution)**: With the boundary closed, you think. You simulate, plan, consider alternatives. No information flows in or out. This is like the isentropic expansion or compression in a thermodynamic cycle: energy moves around internally, but total entropy doesn't change.

3. **Action (Expansion)**: You take your low-entropy intention and expand it into a high-entropy motor command---all those fine details of muscle activations and motor signals. This is like the power stroke of an engine: you're doing work on the external world.

And just like a heat engine, there's an efficiency bound. The Carnot limit tells you how well you can convert thermal energy to mechanical work. Here, the analog tells you how well you can convert sensory information to control information. Perfection is impossible---some "waste heat" (irreversible information loss) is inevitable.
:::

The agent's interaction loop is a **belief density evolution cycle** on the information manifold.

:::{prf:definition} Cycle Phases
:label: def-cycle-phases


| Phase             | Process            | Information Flow                      | Entropy Change               |
|-------------------|--------------------|---------------------------------------|------------------------------|
| **I. Perception** | Compression        | Mutual information $I(X;K)$ extracted | $\Delta S_{\text{bulk}} < 0$ |
| **II. Dreaming**  | Internal evolution | No external exchange                  | $\Delta S = 0$ (isentropic)  |
| **III. Action**   | Expansion          | Mutual information $I(A;K)$ injected  | $\Delta S_{\text{bulk}} > 0$ |

*Remark (Statistical mechanics analogy).* This cycle is structurally analogous to a Stirling cycle in thermodynamics.

:::
:::{prf:theorem} Perception as Compression
:label: thm-perception-as-compression

During perception, the agent compresses external entropy into internal free energy:

$$
W_{\text{compress}} = T_c \cdot I(X_t; K_t) \geq 0,

$$
where $T_c$ is the cognitive temperature ({prf:ref}`def-cognitive-temperature`) and $I(X_t; K_t)$ is the mutual information extracted from the observation $X_t$ into the macro-state $K_t$.

*Mechanism:* The Visual Encoder $E_\phi$ compresses high-entropy raw data $\phi_{\text{raw}}$ into a low-entropy macro-state $z$. The "heat" absorbed is the raw sensory stream.

*Information-theoretic interpretation:* Entropy decreases ($\Delta S < 0$). The Information Bottleneck cost bounds the compression.

:::
:::{prf:theorem} Action as Expansion
:label: thm-action-as-expansion

During action, the agent expands internal free energy into external control:

$$
W_{\text{expand}} = T_c \cdot I(K^{\text{act}}_t; K_t) \geq 0,

$$
where $I(K^{\text{act}}_t; K_t)$ is the mutual information injected from the intention into the motor output.

*Mechanism:* The Action Decoder $D_A$ "expands" the low-entropy Intention $u_\pi$ into high-dimensional motor commands $a_{\text{raw}}$, injecting motor texture.

*Information-theoretic interpretation:* Entropy increases ($\Delta S > 0$). The agent injects stochastic texture into motor outputs.

:::
:::{prf:definition} Dreaming as Unitary Evolution
:label: def-dreaming-as-unitary-evolution

In the dreaming phase, the internal dynamics are approximately unitary (energy-conserving):

$$
\partial_s \rho + [H_{\text{internal}}, \rho]_{\text{Poisson}} = 0,

$$
where $H_{\text{internal}}$ is the effective Hamiltonian:

$$
H_{\text{internal}}(z, p) = \frac{1}{2}\|p\|_{G^{-1}}^2 + V_{\text{critic}}(z).

$$
*Mechanism:* The agent is decoupled from the boundary (adiabatic/isolated). The Bulk evolves under Hamiltonian dynamics (BAOAB integrator with $\gamma \to 0$).

*Information-theoretic interpretation:* Isentropic ($\Delta S = 0$). Internal planning proceeds without information exchange with the environment.

:::
:::{prf:proposition} Carnot Efficiency Bound
:label: prop-carnot-efficiency-bound

The agent's efficiency in converting sensory information to control information is bounded:

$$
\eta = \frac{I(K^{\text{act}}_t; K_t)}{I(X_t; K_t)} \leq 1 - \frac{T_{\text{motor}}}{T_{\text{sensor}}},

$$
where $T_{\text{sensor}}$ and $T_{\text{motor}}$ are the effective temperatures at the sensory and motor boundaries.

*Interpretation:* Perfect efficiency ($\eta = 1$) requires $T_{\text{motor}} = 0$ (deterministic motors) or $T_{\text{sensor}} \to \infty$ (infinite sensory entropy). Real systems operate at $\eta < 1$.

**Cross-references:** {ref}`sec-adaptive-thermodynamics` (Adaptive Thermodynamics), {ref}`sec-the-equivalence-theorem` (MaxEnt Control).

*Forward reference (Reward as Heat).* {ref}`sec-the-bulk-potential-screened-poisson-equation` establishes that Reward is the thermodynamic **heat input** that drives the cycle: the Boltzmann-Value Law (Axiom {prf:ref}`ax-the-boltzmann-value-law`) identifies $V(z) = E(z) - T_c S(z)$ as Gibbs Free Energy, and Theorem {prf:ref}`thm-wfr-consistency-value-creates-mass` proves that WFR dynamics materialize the agent in high-value regions ("Value Creates Mass").

:::
(sec-wfr-boundary-conditions-waking-vs-dreaming)=
## WFR Boundary Conditions: Waking vs Dreaming

:::{div} feynman-prose
Now we get to something philosophically deep: what's the difference between being awake and dreaming? In ordinary language, we might say "when you're awake, your senses are active; when you're dreaming, they're not." But can we make this precise?

Yes, we can. The difference is entirely in the boundary conditions.

When you're awake, your sensors are clamped to reality. Every moment, the world is shouting at you through your eyes and ears, forcing your belief state to match what's out there. That's a Dirichlet condition---the boundary is fixed to external observations.

When you're dreaming, the boundary is *reflective*. No information flows in from outside. No information flows out to motors. Your mind is a closed system, evolving under its own internal dynamics. The boundary is sealed.

Mathematically, this is the simplest possible change: swap a Dirichlet condition for a Neumann-zero condition. But the consequences are profound. In waking, your beliefs are constantly being corrected by reality. In dreaming, your beliefs can wander wherever the internal dynamics take them. That's why dreams can be so strange---there's no ground truth pulling you back.
:::

The **Wasserstein-Fisher-Rao** (WFR, {prf:ref}`def-the-wfr-action`) equation from {ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces` governs the belief density $\rho$. The distinction between Waking and Dreaming is rigorously defined by the **boundary condition** on $\rho$. Boundary conditions update at interaction time $t$, while internal flow evolves in computation time $s$ ({ref}`sec-the-chronology-temporal-distinctions`).

:::{prf:definition} Waking: Boundary Clamping
:label: def-waking-boundary-clamping

During waking ($u_\pi \neq 0$), the sensory stream creates a high-mass source at the encoded location:

$$
\rho_{\partial}^{\text{sense}}(z, t) = \delta(z - z_{\text{obs}}(t)) \quad \text{(Dirichlet)},

$$
and the motor stream creates a flux sink:

$$
\nabla_n \rho \cdot \mathbf{n} = j_{\text{motor}}(u_\pi) \quad \text{(Neumann)}.

$$
The internal belief $\rho_{\text{bulk}}$ evolves to minimize the **WFR Geodesic Distance** to $\rho_{\partial}$:
- **Small Error** ($d_{\text{WFR}} < \lambda$): Transport dominates ($v$ term). The agent smoothly tracks the observation.
- **Large Error** ($d_{\text{WFR}} > \lambda$): Reaction dominates ($r$ term). The agent "teleports" (Surprise/Saccade) to the new reality via chart jump.

:::
:::{prf:definition} Dreaming: Reflective Boundary
:label: def-dreaming-reflective-boundary

During dreaming ($u_\pi = 0$), the sensory stream is cut. The boundary condition becomes **Reflective**:

$$
\nabla_n \rho \cdot \mathbf{n} = 0 \quad \text{(Reflective/Neumann-zero)}.

$$
The system is closed:
- Total mass is conserved: $\int_{\mathcal{Z}} \rho\, r\, d\mu_G = 0$
- Dynamics are driven purely by the internal potential $V_{\text{critic}}(z)$
- No information enters or leaves the boundary

:::
:::{prf:theorem} WFR Mode Switching
:label: thm-wfr-mode-switching

The transition from waking to dreaming corresponds to a **boundary condition phase transition**:

| Mode         | Sensory BC                             | Motor BC             | Internal Flow | Information Balance       |
|--------------|----------------------------------------|----------------------|---------------|---------------------------|
| **Waking**   | Dirichlet ($\delta$-clamp)             | Neumann (flux-clamp) | Source-driven | $\oint j_{\text{in}} > 0$ |
| **Dreaming** | Reflective ($\nabla \rho \cdot n = 0$) | Reflective           | Recirculating | $\oint j = 0$             |

:::
:::{prf:proposition} Grounding Rate via Boundary Flux
:label: prop-grounding-rate-via-boundary-flux

The grounding rate (cf. Definition 16.1.1) is:

$$
G_t = \oint_{\partial\mathcal{Z}_{\text{sense}}} j_{\text{obs}} \cdot dA - \oint_{\partial\mathcal{Z}_{\text{motor}}} j_{\text{motor}} \cdot dA,

$$
which is:
- **Positive** during waking (net information inflow from sensors)
- **Zero** during dreaming (closed system)
- **Negative** during pure actuation (net information outflow to motors)

**Cross-references:** {ref}`sec-the-wfr-metric` (WFR Action), {ref}`sec-the-unified-world-model` (WFR World Model), {ref}`sec-the-interface-and-observation-inflow` (Observation Inflow).

:::
(sec-the-context-space-unified-definition)=
## The Context Space: Unified Definition

:::{div} feynman-prose
Now I want to show you something that I find really beautiful---a unification that wasn't obvious at all until we had the right framework.

What do these three things have in common?
- A robot deciding which direction to push a lever
- A classifier deciding whether an image shows a cat or a dog
- A language model deciding which word comes next given a prompt

On the surface, they seem totally different. Actions, labels, tokens---different domains, different vocabularies, different applications. But in the framework we've been building, they're all the same thing: *boundary conditions*.

Each one specifies a constraint on how the agent's internal state should flow outward. The robot's action says "push this way." The classifier's label says "route to this output category." The prompt says "generate text in this direction." In every case, you're clamping the motor boundary to a particular configuration.

We call the space of all such boundary conditions the *Context Space* $\mathcal{C}$. And the remarkable fact is that the mathematics doesn't care which interpretation you use. The bulk dynamics---the geodesic flows, the WFR transport, the holographic generation---all work the same way. Only the boundary semantics change.

This is why the same neural network architectures can be adapted from robotics to language modeling to classification: they're all implementing the same geometric structure with different boundary interpretations.
:::

The Action Atlas admits a deeper structure: the **Context Space** $\mathcal{C}$ is the abstract space of boundary conditions that unifies RL actions, classification labels, and LLM prompts.

:::{prf:definition} Context Space
:label: def-context-space

The **Context Space** $\mathcal{C}$ is a manifold parameterizing the control/conditioning signal for the agent:

$$
\mathcal{C} := \{c : c \text{ specifies a boundary condition on } \partial\mathcal{Z}\}.

$$
The context determines the target distribution at the motor boundary via the effective potential:

$$
\pi(a | z, c) \propto \exp\left(-\frac{1}{T_c} \Phi_{\text{eff}}(z, K, c)\right).

$$
Units: $[\mathcal{C}]$ inherits from the task domain.

:::
:::{prf:definition} Context Instantiation Functor
:label: def-context-instantiation-functor

The Context Space admits a functor $\mathcal{I}: \mathbf{Task} \to \mathcal{C}$ with three canonical instantiations:

| Task Domain        | Context $c \in \mathcal{C}$ | Motor Output $a$           | Effective Potential $\Phi_{\text{eff}}$      |
|--------------------|-----------------------------|----------------------------|----------------------------------------------|
| **RL**             | Action space $\mathcal{A}$  | Motor command (torques)    | $V_{\text{critic}}(z, K)$                    |
| **Classification** | Label space $\mathcal{Y}$   | Class prediction $\hat{y}$ | $-\log p(y\mid z)$ (cross-entropy)           |
| **LLM**            | Prompt space $\mathcal{P}$  | Token sequence             | $-\log p(\text{token}\mid z, \text{prompt})$ |

*Key Insight:* In all cases, the context $c$ functions as the **symmetry-breaking boundary condition** that determines which direction the holographic expansion takes at the origin.

:::
:::{prf:theorem} Universal Context Structure
:label: thm-universal-context-structure

All context instantiations share the same geometric structure:

1. **Embedding:** $c \mapsto e_c \in \mathbb{R}^{d_c}$ maps the context to a latent vector
2. **Symmetry-Breaking Kick:** $e_c$ determines the initial control field:

   $$
   u_\pi(0) = G^{-1}(0) \cdot e_c = \frac{1}{4} e_c

   $$
   (at the Poincare disk origin where $G(0) = 4I$)
3. **Motor Distribution:** The output distribution is:

   $$
   \pi(a | z, c) = \text{softmax}\left(-\frac{\Phi_{\text{eff}}(z, K, c)}{T_c}\right)

   $$
*Proof.* The holographic expansion ({ref}`sec-radial-generation-entropic-drift-and-policy-control`) is invariant to the interpretation of the control field $u_\pi$. Whether $u_\pi$ encodes "go left" (RL), "class = cat" (classification), or "continue with tone = formal" (LLM), the bulk dynamics follow the same geodesic SDE ({ref}`sec-the-equations-of-motion-geodesic-jump-diffusion`). The interpretation is purely a boundary condition. $\square$

:::
:::{prf:definition} Context-Conditioned WFR
:label: def-context-conditioned-wfr

The WFR dynamics ({ref}`sec-the-wfr-metric`) generalize to context-conditioned form:

$$
\partial_s \rho + \nabla \cdot (\rho\, v_c) = \rho\, r_c,

$$
where:
- $v_c(z) = -G^{-1}(z) \nabla_z \Phi_{\text{eff}}(z, K, c) + u_\pi(z, c)$ is the context-conditioned velocity
- $r_c(z)$ is the context-conditioned reaction rate (chart jumps influenced by context)

:::
:::{prf:corollary} Prompt = Action = Label
:label: cor-prompt-action-label

The following are isomorphic as boundary conditions on $\partial\mathcal{Z}$:

$$
\text{RL Action} \;\cong\; \text{Classification Label} \;\cong\; \text{LLM Prompt}.

$$
Each specifies:
1. **Which chart** to route to (discrete macro $K$ or $A$)
2. **Where in the chart** to aim (continuous nuisance $z_n$ or $z_{n,\text{motor}}$)
3. **What texture** to inject (visual or motor texture)

*Remark (Unified Training Objective).* This isomorphism enables transfer learning across task domains: an agent trained on RL can be fine-tuned for classification by reinterpreting the action space as label space, with the same holographic dynamics.

**Cross-references:** {ref}`sec-policy-control-field` (Control Field), Theorem {prf:ref}`thm-unified-control-interpretation`, Definition {prf:ref}`def-effective-potential`.

*Forward reference (Effective Potential Resolution).* {ref}`sec-the-bulk-potential-screened-poisson-equation`
resolves the meaning of $\Phi_{\text{eff}} = V_{\text{critic}}$: the Critic solves the **Screened Poisson Equation** to
compute the potential from boundary reward flux (scalar charges in the conservative case). The discount factor $\gamma$
determines the screening length $\ell = c_{\text{info}} \Delta t / (-\ln\gamma)$ (natural units: $1/(-\ln\gamma)$)
(Corollary {prf:ref}`cor-discount-as-screening-length`),
explaining why distant rewards are exponentially suppressed in policy.

:::
(sec-implementation-the-holographicinterface-module)=
## Implementation: The HolographicInterface Module

:::{div} feynman-prose
All right, enough theory. Let's build something.

The code below implements everything we've discussed: the dual atlas architecture, the motor texture decomposition, the context-conditioned policy. You'll see how the abstract mathematics translates into concrete PyTorch modules.

A few things to notice as you read through:
1. The Visual and Action atlases have parallel structure---this is the Legendre duality made concrete
2. Motor texture is sampled with geometry-dependent variance---the conformal scaling from the Poincare disk
3. The policy is context-conditioned---the same network handles RL actions, classification labels, and more

This isn't just an illustration. This is a working architecture that embodies the boundary interface theory.
:::

We provide the Python implementation of the Holographic Interface, combining the Dual Atlas, Motor Texture, and Context Space.

**Algorithm 23.7.1 (HolographicInterface Module).**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Tuple
from enum import Enum


class BoundaryConditionType(Enum):
    """Definition 23.1.2-23.1.3: Boundary condition types."""
    DIRICHLET = "dirichlet"    # Position clamping (sensors)
    NEUMANN = "neumann"        # Flux clamping (motors)
    REFLECTIVE = "reflective"  # Dreaming mode (zero flux)


class ContextType(Enum):
    """Definition 23.6.2: Context instantiation types."""
    RL = "rl"                        # Action space
    CLASSIFICATION = "classification"  # Label space
    LLM = "llm"                      # Prompt space


@dataclass
class InterfaceConfig:
    """Configuration for HolographicInterface."""
    obs_dim: int = 64
    action_dim: int = 8
    latent_dim: int = 32
    hidden_dim: int = 256
    num_visual_charts: int = 8
    num_action_charts: int = 4
    codes_per_chart: int = 64
    context_dim: int = 64
    sigma_motor: float = 0.1
    T_c: float = 1.0


class DualAtlasEncoder(nn.Module):
    """
    Definitions 23.2.1-23.2.2: Symmetric encoder for Visual/Action atlases.

    Extends AttentiveAtlasEncoder ({ref}`sec-tier-the-attentive-atlas`) with unified interface.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_charts: int,
        codes_per_chart: int,
        atlas_type: Literal["visual", "action"],
    ):
        super().__init__()
        self.atlas_type = atlas_type
        self.num_charts = num_charts
        self.latent_dim = latent_dim

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Cross-attention routing ({ref}`sec-tier-the-attentive-atlas`)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.chart_queries = nn.Parameter(torch.randn(num_charts, hidden_dim) * 0.02)
        self.scale = hidden_dim ** 0.5

        # Per-chart codebooks
        self.codebooks = nn.ModuleList([
            nn.Embedding(codes_per_chart, latent_dim)
            for _ in range(num_charts)
        ])

        # Residual decomposition
        self.nuisance_head = nn.Linear(hidden_dim, latent_dim)
        self.texture_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode input to (macro, nuisance, texture) triple."""
        B = x.shape[0]

        # Feature extraction
        h = self.feature_extractor(x)

        # Cross-attention routing
        k = self.key_proj(h)  # [B, H]
        attn = torch.einsum('bh,ch->bc', k, self.chart_queries) / self.scale
        chart_probs = F.softmax(attn, dim=-1)  # [B, C]
        chart_idx = chart_probs.argmax(dim=-1)  # [B]

        # VQ from selected chart
        z_macro = torch.stack([
            self.codebooks[c.item()](torch.zeros(1, dtype=torch.long, device=x.device)).squeeze()
            for c in chart_idx
        ])  # [B, D]

        # Residual decomposition
        z_nuisance = self.nuisance_head(h)
        z_texture = self.texture_head(h)

        return {
            'chart_idx': chart_idx,
            'chart_probs': chart_probs,
            'z_macro': z_macro,
            'z_nuisance': z_nuisance,
            'z_texture': z_texture,
        }


def sample_motor_texture(
    z: torch.Tensor,
    d_motor_tex: int,
    sigma_motor: float,
) -> torch.Tensor:
    """
    Definition 23.3.3: Sample motor texture with conformal scaling.

    Sigma_motor(z) = sigma^2 * G^{-1}(z) = sigma^2 * (1-|z|^2)^2 / 4
    """
    B = z.shape[0]
    device = z.device

    # Conformal factor at z (Poincare disk)
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    G_inv_scale = (1.0 - r_sq.clamp(max=0.99)) ** 2 / 4.0

    # Sample with geometry-dependent variance
    xi = torch.randn(B, d_motor_tex, device=device)
    z_tex_motor = sigma_motor * torch.sqrt(G_inv_scale) * xi

    return z_tex_motor


class ContextConditionedPolicy(nn.Module):
    """
    Definition 23.6.4: Context-conditioned policy for unified task handling.

    Unifies RL actions, classification labels, and LLM tokens.
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Context embedding
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        T_c: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute context-conditioned action distribution.

        pi(a|z, c) = softmax(-Phi_eff(z, K, c) / T_c)
        """
        # Embed context
        c_embed = self.context_encoder(context)

        # Concatenate state and context
        z_c = torch.cat([z, c_embed], dim=-1)

        # Compute logits (negative effective potential)
        logits = self.policy_net(z_c)

        # Softmax with temperature
        probs = F.softmax(logits / T_c, dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'context_embedding': c_embed,
        }


class HolographicInterface(nn.Module):
    """
    {ref}`sec-the-boundary-interface-symplectic-structure`: The Holographic Interface.

    Implements the symplectic boundary between Agent and Environment.
    Combines:
    - Dual Atlas (Visual + Action)
    - Motor Texture sampling
    - Context-conditioned policy
    - Thermodynamic cycle tracking

    Cross-references:
    - {ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces` (WFR Geometry)
    - {ref}`sec-radial-generation-entropic-drift-and-policy-control` (Holographic Generation {cite}`thooft1993holographic,susskind1995world`)
    - {ref}`sec-the-equations-of-motion-geodesic-jump-diffusion` (Geodesic SDE)
    """

    def __init__(self, config: InterfaceConfig):
        super().__init__()
        self.config = config

        # Visual Atlas (Definition 23.2.1)
        self.visual_atlas = DualAtlasEncoder(
            config.obs_dim, config.hidden_dim, config.latent_dim,
            config.num_visual_charts, config.codes_per_chart, "visual"
        )

        # Action Atlas (Definition 23.2.2)
        self.action_atlas = DualAtlasEncoder(
            config.action_dim, config.hidden_dim, config.latent_dim,
            config.num_action_charts, config.codes_per_chart, "action"
        )

        # Context-conditioned policy (Definition 23.6.4)
        self.policy = ContextConditionedPolicy(
            config.latent_dim, config.context_dim,
            config.action_dim, config.hidden_dim
        )

        # Action decoder (tangent bundle decoder)
        self.action_decoder = nn.Sequential(
            nn.Linear(config.latent_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

    def forward_perception(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase I: Compression (Environment -> Bulk).
        Thermodynamics: Entropy reduction, heat release.
        Implements Definition 23.1.2 (Dirichlet BC).
        """
        return self.visual_atlas(x)

    def forward_actuation(
        self,
        z: torch.Tensor,
        u_intent: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase III: Expansion (Bulk -> Environment).
        Thermodynamics: Entropy increase, work done.
        Implements Definition 23.1.3 (Neumann BC).
        """
        B = z.shape[0]

        # Sample motor texture (Definition 23.3.3)
        z_tex_motor = sample_motor_texture(
            z, self.config.latent_dim, self.config.sigma_motor
        )

        # Decode intention to action
        z_u = torch.cat([z, u_intent], dim=-1)
        a_base = self.action_decoder(z_u)

        # Add motor texture
        a_raw = a_base + z_tex_motor[:, :self.config.action_dim]

        return {
            'action': a_raw,
            'action_base': a_base,
            'motor_texture': z_tex_motor,
        }

    def forward_proprioception(
        self,
        a_realized: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Inverse model: Map realized actions back to latent intentions.
        Used to calculate execution error.
        """
        return self.action_atlas(a_realized)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mode: Literal["waking", "dreaming"] = "waking",
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through holographic interface.

        Args:
            x: Observation [B, obs_dim]
            context: Context conditioning [B, context_dim]
            mode: "waking" (boundary clamped) or "dreaming" (reflective)
        """
        # Phase I: Perception (compression)
        vis_out = self.forward_perception(x)
        z = vis_out['z_nuisance']  # Use nuisance as state

        # Get context-conditioned policy
        policy_out = self.policy(z, context, self.config.T_c)

        if mode == "dreaming":
            # Reflective boundary: no actuation
            return {
                'visual': vis_out,
                'policy': policy_out,
                'mode': mode,
            }

        # Phase III: Action (expansion)
        u_intent = policy_out['context_embedding']
        act_out = self.forward_actuation(z, u_intent, context)

        return {
            'visual': vis_out,
            'policy': policy_out,
            'action': act_out,
            'mode': mode,
        }
```

**Cross-references:** {ref}`sec-tier-the-attentive-atlas` (AttentiveAtlasEncoder), {ref}`sec-decoder-architecture-overview-topological-decoder` (TopologicalDecoder), Algorithm 22.4.2 (BAOAB).

::::{admonition} Connection to RL #7: Dreamer/World Models as Generic RNN Dynamics
:class: note
:name: conn-rl-7
**The General Law (Fragile Agent):**
The HolographicInterface implements latent dynamics via **symplectic integrators** on the state-space manifold $(\mathcal{Z}, G, \omega)$. The BAOAB algorithm ({ref}`sec-the-geodesic-baoab-integrator`) preserves the symplectic structure:

$$
\hat{z}_{t+1} = \Phi_{\text{BAOAB}}(z_t) \quad \text{with} \quad \omega(\Phi_* X, \Phi_* Y) = \omega(X, Y).

$$
The dual atlas structure (Definition 23.2) decomposes latent space into Visual and Action atlases with matched boundary conditions.

**The Degenerate Limit:**
Remove symplectic structure: replace $\Phi_{\text{BAOAB}}$ with generic RNN/GRU. Ignore boundary condition matching.

**The Special Case (Standard RL - Dreamer, MuZero):**
World-model RL uses generic neural network dynamics:

$$
z_{t+1} = f_\theta(z_t, a_t), \quad \hat{r}_t = r_\theta(z_t), \quad \hat{\gamma}_t = \gamma_\theta(z_t).

$$
The RNN/GRU/Transformer architecture has no geometric constraints—it's a universal function approximator.

**Result:** Dreamer/MuZero/PlaNet are the $\omega \to 0$ limit where symplectic structure is ignored. The agent learns arbitrary dynamics rather than energy-conserving flows.

**What the generalization offers:**
- **Conservation guarantees**: Symplectic integrators preserve phase-space volume (Liouville's theorem)
- **Long-horizon stability**: Energy drift bounded by $O((\Delta t)^k)$ for $k$th-order integrators
- **Interpretable rollouts**: Latent trajectories follow geodesics modified by potential forces
- **Boundary semantics**: Dirichlet/Neumann conditions distinguish observation vs action interfaces (Definition 23.1)
::::

(sec-summary-tables-and-diagnostic-nodes-a)=
## Summary Tables and Diagnostic Nodes

**Summary of Holographic Interface:**

| Component              | Visual (Perception)           | Motor (Action)                  |
|------------------------|-------------------------------|---------------------------------|
| **Boundary Condition** | Dirichlet (position clamp)    | Neumann (flux clamp)            |
| **Atlas**              | $\mathcal{A}_{\text{vis}}$    | $\mathcal{A}_{\text{act}}$      |
| **Macro**              | Chart index $K$               | Action primitive $K^{\text{act}}$ |
| **Nuisance**           | Pose/viewpoint $z_n$          | Compliance $z_{n,\text{motor}}$ |
| **Texture**            | Pixel detail $z_{\text{tex}}$ | Tremor $z_{\text{tex,motor}}$   |
| **Thermodynamics**     | Compression ($\Delta S < 0$)  | Expansion ($\Delta S > 0$)      |

**Context Space Instantiation:**

| Task           | Context $c$  | Output          | Potential $\Phi_{\text{eff}}$              |
|----------------|--------------|-----------------|--------------------------------------------|
| RL             | Action space | Motor command   | $V_{\text{critic}}$                        |
| Classification | Label space  | Class $\hat{y}$ | $-\log p(y\mid z)$                         |
| LLM            | Prompt space | Token           | $-\log p(\text{tok}\mid z, \text{prompt})$ |

(node-30)=
**Node 30: SymplecticBoundaryCheck**

| **#**  | **Name**                    | **Component** | **Type**           | **Interpretation**               | **Proxy**                                                | **Cost** |
|--------|-----------------------------|---------------|--------------------|----------------------------------|----------------------------------------------------------|----------|
| **30** | **SymplecticBoundaryCheck** | **Interface** | **BC Consistency** | Are sensor/motor BCs compatible? | $\lVert\omega(j_{\text{sense}}, j_{\text{motor}})\rVert$ | $O(Bd)$  |

**Trigger conditions:**
- High SymplecticBoundaryCheck: Sensor and motor boundary conditions violate symplectic structure.
- Remedy: Recalibrate boundary coupling; verify Legendre transform consistency; check phase space constraints.

(node-31)=
**Node 31: DualAtlasConsistencyCheck**

| **#**  | **Name**                      | **Component** | **Type**          | **Interpretation**                     | **Proxy**                                                                  | **Cost**  |
|--------|-------------------------------|---------------|-------------------|----------------------------------------|----------------------------------------------------------------------------|-----------|
| **31** | **DualAtlasConsistencyCheck** | **Encoder**   | **Atlas Duality** | Are Visual and Action atlases aligned? | $\lVert e_\alpha^{\text{vis}} - \mathcal{L}(e_\beta^{\text{act}})\rVert^2$ | $O(BK^2)$ |

**Trigger conditions:**
- High DualAtlasConsistencyCheck: Visual and Action atlases have drifted apart.
- Remedy: Increase Legendre alignment loss; verify codebook coupling; check chart transition consistency.

(node-32)=
**Node 32: MotorTextureCheck**

| **#**  | **Name**              | **Component** | **Type**           | **Interpretation**                       | **Proxy**                                                  | **Cost**               |
|--------|-----------------------|---------------|--------------------|------------------------------------------|------------------------------------------------------------|------------------------|
| **32** | **MotorTextureCheck** | **Policy**    | **Motor Firewall** | Is motor texture decoupled from control? | $\lVert\partial_{z_{\text{tex,motor}}} \pi(a\mid z)\rVert$ | $O(Bd_{\text{motor}})$ |

**Trigger conditions:**
- High MotorTextureCheck: Motor texture is leaking into control decisions (firewall violated).
- Remedy: Increase motor texture firewall penalty; verify motor residual decomposition; check Axiom {prf:ref}`ax-motor-texture-firewall`.

(node-33)=
**Node 33: ThermoCycleCheck**

| **#**  | **Name**             | **Component**   | **Type**           | **Interpretation**                               | **Proxy**                                              | **Cost** |
|--------|----------------------|-----------------|--------------------|--------------------------------------------------|--------------------------------------------------------|----------|
| **33** | **ThermoCycleCheck** | **World Model** | **Energy Balance** | Is perception/action thermodynamically balanced? | $\lvert W_{\text{compress}} - W_{\text{expand}}\rvert$ | $O(B)$   |

**Trigger conditions:**
- High ThermoCycleCheck: Thermodynamic imbalance between perception and action phases.
- Remedy: Recalibrate information flow rates; verify boundary coupling; check Carnot efficiency bound (Proposition {prf:ref}`prop-carnot-efficiency-bound`).

(node-34)=
**Node 34: ContextGroundingCheck**

| **#**  | **Name**                  | **Component** | **Type**             | **Interpretation**                          | **Proxy**                 | **Cost** |
|--------|---------------------------|---------------|----------------------|---------------------------------------------|---------------------------|----------|
| **34** | **ContextGroundingCheck** | **Policy**    | **Context Validity** | Is context properly grounding motor output? | $I(K^{\text{act}}_t; c) / I(X_t; K_t)$ | $O(B)$   |

**Trigger conditions:**
- Low ContextGroundingCheck: Context is not influencing motor output (ungrounded generation).
- Remedy: Increase context embedding strength; verify context-conditioned potential; check symmetry-breaking kick.
