(sec-the-equations-of-motion-geodesic-jump-diffusion)=
# The Equations of Motion: Geodesic Jump-Diffusion

## TLDR

- Once geometry is fixed, dynamics “almost write themselves”: the agent follows **controlled geodesic motion** with
  stochastic exploration noise.
- Continuous motion within charts + discrete chart transitions yields a **jump-diffusion** equation of motion.
- The metric plays the role of **mass/preconditioner**: curvature slows or speeds updates depending on capacity and risk.
- This chapter is the bridge from geometric laws (metric/WFR) to implementable update rules and diagnostics.
- Use the diagnostic suite (geodesic consistency, jump consistency, boundary-reached checks) to validate dynamics.

## Roadmap

1. Continuous-time limit: geodesic/Langevin-style motion on $(\mathcal{Z},G)$.
2. Discrete events: jump rates and chart transitions.
3. Summary tables and diagnostics for implementation validation.

{cite}`oksendal2003sde,risken1996fokkerplanck`

(rb-continuous-actor-critic)=
:::{admonition} Researcher Bridge: Continuous-Time Actor-Critic
:class: info
The equations of motion are the continuous-time limit of policy updates with stochastic exploration noise. Think of it as a Langevinized actor-critic where the metric defines the preconditioner.
:::

:::{div} feynman-prose
Now, here is where everything comes together. We have built up all this machinery in the previous sections---the geometry, the metric law, the WFR transport---and you might be wondering: what does the agent actually *do*? How does it move?

This is the chapter where we write down the answer. And the beautiful thing is that once you have the geometry right, the equations of motion almost write themselves. They are not something we impose from outside; they *emerge* from the structure we have already established.

The key insight is this: **the agent's motion is geodesic motion on a curved manifold, with noise**. That is it. The agent follows the paths of least resistance through the latent space, but it gets jostled around by thermal fluctuations. And occasionally---this is the jump part---it can hop from one chart to another when it discovers a better representation.

If you have ever studied classical mechanics, this should feel familiar. We are doing Lagrangian mechanics, but in a stochastic setting and on a curved space. The curvature comes from the metric law of Section 18. The noise comes from the exploration-exploitation tradeoff. And the jumps come from the discrete structure of the chart atlas.

Let me emphasize something that might seem obvious but is actually profound: **the metric plays the role of mass**. In ordinary mechanics, mass tells you how hard it is to accelerate an object. Here, the metric tells you how hard it is to move through a region of latent space. High-curvature regions are "heavy"---the agent slows down there, takes smaller steps, is more cautious. Low-curvature regions are "light"---the agent moves freely.

Why is that the right thing? Because high curvature means high risk, high uncertainty, regions where the representation is strained. You want to be careful there. The geometry *is* the caution.
:::

We derive the rigorous equation of motion (EoM) for the agent. This equation unifies the WFR geometry ({ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`, {prf:ref}`def-the-wfr-action`), the Metric Law ({ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`, Theorem {prf:ref}`thm-capacity-constrained-metric-law`), and the Policy-driven Expansion ({ref}`Section 21 <sec-radial-generation-entropic-drift-and-policy-control>`).

(sec-the-stochastic-action-principle)=
## The Stochastic Action Principle (Mass = Metric)

:::{div} feynman-prose
Before we get into the formalism, let me tell you what we are really doing here. In classical mechanics, you have the action principle: Nature chooses the path that minimizes the action integral. But what does "minimize" mean when there is randomness involved?

The answer is the Onsager-Machlup functional. Instead of asking "which path does the particle take?", we ask "which path is *most probable*?" And it turns out that the most probable path is the one that minimizes a certain action---not the classical action, but a modified one that accounts for the noise.

Here is the picture I want you to have in your mind. Imagine you are looking at a particle undergoing Brownian motion in a potential. At any moment, the particle could go anywhere---the noise is pushing it in all directions. But some paths are more likely than others. The Onsager-Machlup functional tells you exactly how likely each path is: $P[\text{path}] \propto \exp(-\text{Action}/T)$, where $T$ is the temperature.

This is exactly the path-integral formulation of statistical mechanics. And the beautiful thing is that it works on curved spaces too, with one subtlety: on a curved manifold, you need a curvature correction term. The measure for path integration is not uniform---it gets distorted by the geometry.

Now, the key decision we have to make: what plays the role of mass? In ordinary mechanics, mass appears in the kinetic energy term, $\frac{1}{2}mv^2$. On a Riemannian manifold, the natural generalization is $\frac{1}{2}G_{ij}\dot{z}^i\dot{z}^j$---the metric-weighted norm of the velocity.

So here is our choice, and it is the conceptually correct one: **Mass = Metric**. The inertial mass tensor *is* the metric tensor. This is not arbitrary; it is forced on us by the geometry. The metric already encodes how to measure distances. It should also encode how to measure kinetic energy.
:::

The classical Lagrangian approach extends to stochastic systems via the **Onsager-Machlup functional**, which assigns a probability to paths based on their "action."

:::{prf:definition} Mass Tensor
:label: def-mass-tensor

We define the **inertial mass tensor** $\mathbf{M}(z)$ as the capacity-constrained metric:

$$
\mathbf{M}(z) := G(z).

$$
This definition has the following operational consequences:
- **High curvature regions** (large $G$) have larger effective mass, yielding smaller velocity updates per unit force
- **Low curvature regions** (small $G$) have smaller effective mass, yielding larger velocity updates per unit force

Units: $[\mathbf{M}_{ij}] = [z]^{-2}$ (same as metric).

*Remark (Risk-Metric Coupling).* Combined with the Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`), this yields a causal chain:

$$
\text{High risk } T_{ij} \;\Rightarrow\; \text{Large } G_{ij} \;\Rightarrow\; \text{Large } \mathbf{M}_{ij} \;\Rightarrow\; \text{Reduced step size}

$$
The metric-weighted step size decreases in high-curvature (high-risk) regions without explicit penalty terms.

:::

:::{admonition} The Causal Chain of Caution
:class: feynman-added tip

The Mass = Metric principle creates an automatic feedback loop for safe exploration:

1. **High risk** in a region $\Rightarrow$ Metric Law says curvature increases $\Rightarrow$ metric $G$ grows
2. **Large metric** $\Rightarrow$ Mass = Metric says effective mass increases
3. **Large mass** $\Rightarrow$ Same force produces smaller acceleration $\Rightarrow$ smaller step size

No explicit "safety penalty" needed---the geometry *is* the safety mechanism.
:::

:::{prf:definition} Extended Onsager-Machlup Action
:label: def-extended-onsager-machlup-action

Let $(\mathcal{Z}, G)$ be the latent Riemannian manifold with the capacity-constrained metric ({ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`). For a path $z: [0, T] \to \mathcal{Z}$, the extended Onsager-Machlup action is:

$$
S_{\mathrm{OM}}[z] = \int_0^T \left( \frac{1}{2}\mathbf{M}(z)\|\dot{z}\|^2 + \Phi_{\text{eff}}(z) + \frac{T_c}{12}\,R(z) + T_c \cdot H_{\pi}(z) \right) ds,

$$
where:
- $\mathbf{M}(z)\|\dot{z}\|^2 = G_{ij}(z)\,\dot{z}^i\,\dot{z}^j$ is the kinetic energy (mass = metric)
- $\Phi_{\text{eff}}(z)$ is the effective potential (Definition {prf:ref}`def-effective-potential`)
- $R(z)$ is the scalar curvature of the metric $G$
- $H_{\pi}(z) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|z)]$ is the policy entropy
- $T_c > 0$ is the {prf:ref}`def-cognitive-temperature` (cf. {ref}`Section 21.2 <sec-policy-control-field>`)

Units: $[S_{\mathrm{OM}}] = \mathrm{nat}$.

*Remark (Curvature Correction).* The term $\frac{T_c}{12}R(z)$ is a stochastic correction that accounts for the path-measure distortion on curved spaces. In flat space ($R = 0$), this term vanishes. The entropy term $T_c H_{\pi}$ ensures the agent prefers stochastic policies in uncertain regions.

:::

:::{div} feynman-prose
Let me decode this action functional term by term, because each piece is doing something important:

**The kinetic term** $\frac{1}{2}G_{ij}\dot{z}^i\dot{z}^j$: This is "how fast am I moving?" in the curved geometry. Not Euclidean speed, but speed measured with the metric. If the metric is large, the same coordinate velocity costs more action.

**The potential term** $\Phi_{\text{eff}}$: This is "where do I want to go?" We will unpack this later, but it combines the hyperbolic expansion drive with the learned value function and a risk penalty.

**The curvature correction** $\frac{T_c}{12}R$: This is subtle. When you do path integrals on a curved manifold, you have to be careful about how you measure paths. The factor of 1/12 is not arbitrary---it comes from the mathematical analysis of the path measure. On a flat space this vanishes, so you can ignore it for Euclidean intuition.

**The entropy term** $T_c H_\pi$: This is the exploration bonus. The agent *prefers* to have a stochastic policy---it gets rewarded for keeping its options open. The temperature $T_c$ controls how much this matters.

Now, why does the most probable path minimize this action? Think of it this way: every path has an associated "cost." The kinetic term penalizes going fast. The potential term penalizes being in bad places. The entropy term rewards keeping options open. Nature---or rather, the stochastic dynamics---samples paths according to how costly they are, preferring low-cost paths exponentially.
:::

(pi-onsager-machlup)=
::::{admonition} Physics Isomorphism: Onsager-Machlup Action
:class: note

**In Physics:** The Onsager-Machlup functional assigns probability to paths in stochastic thermodynamics: $P[\gamma] \propto \exp(-S_{OM}[\gamma]/k_B T)$ where $S_{OM}$ includes kinetic and potential terms plus a curvature correction {cite}`onsager1953fluctuations`.

**In Implementation:** The extended Onsager-Machlup action (Definition {prf:ref}`def-extended-onsager-machlup-action`):

$$
S_{\text{OM}}[z] = \int_0^T \left(\frac{1}{2}G_{ij}\dot{z}^i\dot{z}^j + \Phi_{\text{eff}} + \frac{T_c}{12}R + T_c H_\pi\right)ds

$$
**Correspondence Table:**

| Statistical Mechanics | Agent (Path Integral) |
|:----------------------|:----------------------|
| Temperature $k_B T$ | Cognitive temperature $T_c$ |
| Kinetic energy $\frac{1}{2}m\lvert\dot{x}\rvert^2$ | $\frac{1}{2}G_{ij}\dot{z}^i\dot{z}^j$ (mass = metric) |
| Potential $U(x)$ | Effective potential $\Phi_{\text{eff}}$ |
| Curvature correction $\frac{k_BT}{12}R$ | $\frac{T_c}{12}R$ |
| Boltzmann weight $e^{-S/k_BT}$ | Path probability $e^{-S_{\text{OM}}/T_c}$ |
::::

:::{div} feynman-prose
Now here is a wonderful consistency check. What happens near the boundary of the Poincare disk? Remember, the boundary represents the "edge" of the representable world---where the agent runs out of capacity.
:::

:::{prf:proposition} Mass Scaling Near Boundary
:label: prop-mass-scaling-near-boundary

For the Poincare disk, the mass tensor scales as:

$$
\mathbf{M}(z) = \frac{4}{(1-|z|^2)^2} I_d \quad \xrightarrow{|z| \to 1} \quad +\infty.

$$
The metric diverges as $|z| \to 1$, which bounds all finite-action trajectories to the interior of the disk.

*Proof.* Direct evaluation of the Poincare metric. The factor $(1-|z|^2)^{-2}$ diverges as $|z| \to 1$. $\square$

:::

:::{div} feynman-prose
This is exactly what we want. The mass becomes infinite at the boundary. What does infinite mass mean? It means the agent *cannot* cross the boundary. No matter how much force you apply, an infinite mass cannot accelerate. The geometry itself enforces the boundary condition---no extra constraint needed.

This is the geometric version of a hard wall. But it is much better than just saying "stop at the boundary," because the agent feels the wall coming. As it approaches the boundary, it gets heavier and heavier, slower and slower. It is not a sudden stop; it is a gradual deceleration. The mathematics is smooth even though the boundary is impenetrable.
:::

:::{prf:proposition} Most Probable Path
:label: prop-most-probable-path

For the controlled diffusion

$$
dz^k = b^k(z)\,ds + \sqrt{2T_c}\,\sigma^{kj}(z)\,dW^j_s,

$$
where $\sigma \sigma^T = G^{-1}$, the most probable path connecting $z(0) = z_0$ and $z(T) = z_1$ minimizes the Onsager-Machlup action $S_{\mathrm{OM}}[z]$ subject to the boundary conditions.

*Proof sketch.* This follows from the Girsanov theorem and the Cameron-Martin formula adapted to Riemannian manifolds. See {cite}`ikeda1989stochastic` Chapter V or {ref}`Appendix A.4 <sec-appendix-a-full-derivations>` for details. $\square$

:::

(sec-the-coupled-jump-diffusion-sde)=
## The Coupled Jump-Diffusion SDE

:::{div} feynman-prose
Alright, now we get to the actual equation of motion. This is where the rubber meets the road.

The agent is not just a point in space. It is a **particle with mass**---and by "mass" here I mean the importance weight $m$, which is like the probability that this particular trajectory is the right one. You can think of it as how much "belief" the agent has invested in this particular path through latent space.

The dynamics have two parts:
1. **Continuous motion**: The particle slides around on the manifold, pulled by gradients and pushed by noise
2. **Discrete jumps**: Occasionally, the particle teleports from one chart to another

Why jumps? Because the latent space is not a single connected manifold. It is an atlas of overlapping charts, like the pages of a flip-book. Sometimes the best move is not to take a small step, but to flip to a completely different page---a different representation, a different conceptual framework.

Think of it like this: you are solving a problem, and you have been thinking about it one way, taking small incremental steps. Then suddenly you realize there is a completely different way to look at it. That is a jump. The continuous dynamics handle the incremental thinking; the jump process handles the conceptual leaps.
:::

The agent's state is not merely a point $z$ but a **particle with mass** $(z, m)$, where $m$ is the importance weight (belief probability). The dynamics couple continuous transport with discrete topological jumps.

*Cross-reference (WFR Boundary Conditions).* The SDE below assumes **Waking mode** boundary conditions (Definition {prf:ref}`def-waking-boundary-clamping`): Dirichlet on sensors (clamping observed position), Neumann on motors (clamping output flux). In **Dreaming mode** (Definition {prf:ref}`def-dreaming-reflective-boundary`), both boundaries become reflective and the flow recirculates internally without external grounding. See {ref}`Section 23.5 <sec-wfr-boundary-conditions-waking-vs-dreaming>` for the mode-switching table and the thermodynamic interpretation ({ref}`Section 23.4 <sec-the-belief-evolution-cycle-perception-dreaming-action>`).

:::{prf:definition} Second-Order Geodesic Langevin Equation
:label: def-bulk-drift-continuous-flow

The agent's state evolves as a **particle with position $z$ and momentum $p$** on the Riemannian manifold $(\mathcal{Z}, G)$. The dynamics are given by the coupled **second-order Langevin SDE**:

$$
\begin{cases}
dz^k = G^{kj}(z)\, p_j\, ds \\[8pt]
dp_k = \left[ -\partial_k \Phi_{\text{eff}} - \gamma\, p_k + \beta_{\text{curl}}\, \mathcal{F}_{kj}\, G^{j\ell}\, p_\ell - \Gamma^m_{k\ell}\, G^{\ell j}\, p_j\, p_m + u_{\pi,k} \right] ds + \sqrt{2\gamma T_c}\, (G^{1/2})_{kj}\, dW^j_s
\end{cases}

$$
where:
- $\Phi_{\text{eff}}$ is the **effective potential** (Definition {prf:ref}`def-effective-potential`)
- $\gamma > 0$ is the **friction coefficient** (damping rate)
- $\mathcal{F}_{ij} = \partial_i \mathcal{R}_j - \partial_j \mathcal{R}_i$ is the **Value Curl** tensor (Definition {prf:ref}`def-value-curl`)
- $\beta_{\text{curl}} \ge 0$ is the **curl coupling strength** (dimensionless)
- $\Gamma^m_{k\ell}$ are the **Christoffel symbols** of the Levi-Civita connection (Proposition {prf:ref}`prop-explicit-christoffel-symbols-for-poincare-disk`)
- $u_{\pi,k}$ is the **control field** from the policy (Definition {prf:ref}`def-the-control-field`)
- $T_c$ is the **cognitive temperature** (Definition {prf:ref}`def-cognitive-temperature`)
- $W_s$ is a standard Wiener process

*Units:* $[z] = \text{length}$, $[p] = \text{length}/\tau$, $[\gamma] = 1/\tau$, $[\Phi_{\text{eff}}] = \mathrm{nat}$, $[T_c] = \mathrm{nat}$.

**Interpretation:** The position evolves via the momentum (kinematic relation), while the momentum evolves under:

1. **Gradient force**: $-\nabla\Phi_{\text{eff}}$ — force from effective potential
2. **Friction**: $-\gamma p$ — damping toward equilibrium
3. **Lorentz force**: $\beta_{\text{curl}} \mathcal{F} G^{-1} p$ — velocity-dependent force from Value Curl (perpendicular to velocity)
4. **Geodesic correction**: $-\Gamma(G^{-1}p, G^{-1}p)$ — parallel transport on curved space
5. **Control field**: $u_\pi$ — policy-induced force
6. **Thermal noise**: $\sqrt{2\gamma T_c} G^{1/2} dW$ — fluctuation-dissipation balanced noise

**Hamiltonian Structure:** The deterministic part ($T_c = 0$, $\gamma = 0$) derives from the Hamiltonian:

$$
H(z, p) = \frac{1}{2} G^{ij}(z)\, p_i\, p_j + \Phi_{\text{eff}}(z).

$$
The friction and noise terms implement an **Ornstein-Uhlenbeck thermostat** that samples the Boltzmann distribution $\rho(z,p) \propto \exp(-H(z,p)/T_c)$.

**Conservative Limit:** When $\mathcal{F} = 0$ (Definition {prf:ref}`def-conservative-reward-field`), the Lorentz term vanishes and we recover the standard geodesic Langevin equation.

**Non-Conservative Dynamics:** When $\mathcal{F} \neq 0$, the Lorentz force induces rotational dynamics. Trajectories may converge to limit cycles rather than fixed points (Theorem {prf:ref}`thm-ness-existence`).

*Remark (BAOAB Integration).* This second-order system is integrated using the Boris-BAOAB scheme (Definition {prf:ref}`def-baoab-splitting`), which preserves the Boltzmann distribution to $O(h^2)$ and handles the velocity-dependent Lorentz force via the Boris rotation.

:::

:::{div} feynman-prose
Let me break down this second-order system, because it looks intimidating but each piece has a clear physical meaning.

**Why two equations?** The agent has both *position* $z$ (where it is) and *momentum* $p$ (how fast it is moving and in what direction). This is like a ball rolling on a curved surface: you need to track both where it is and how fast it is going.

**The position equation** $dz = G^{-1}p\, ds$: This just says "position changes according to velocity." The $G^{-1}$ converts momentum (a covector) to velocity (a vector) using the metric.

**The momentum equation** has several forces:

**The gradient term** $-\nabla\Phi_{\text{eff}}$: This is "roll downhill." The agent feels a force pushing it toward lower potential.

**The friction term** $-\gamma p$: This is damping. Without it, the agent would coast forever. Friction brings it toward equilibrium, balancing against the thermal noise.

**The control term** $u_\pi$: This is the policy. The agent can choose to go somewhere that is not just downhill. The policy provides intentional override of the gradient.

**The Lorentz force** $\beta_{\text{curl}} \mathcal{F} G^{-1} p$: When the reward field has curl, the agent feels a sideways force perpendicular to its velocity---exactly like a charged particle in a magnetic field. It makes the agent spiral or orbit rather than just fall to the bottom.

**The geodesic correction** $-\Gamma(v, v)$ where $v = G^{-1}p$: On a curved manifold, straight lines curve. This term corrects for that, keeping the trajectory on the manifold.

**The thermal noise** $\sqrt{2\gamma T_c} G^{1/2} dW$: Exploration via random thermal fluctuations. The $\sqrt{2\gamma T_c}$ factor ensures **fluctuation-dissipation balance**: friction and noise are calibrated so the system samples the correct Boltzmann distribution.

The beautiful thing is that these are not six separate mechanisms bolted together. They all emerge from the same underlying structure: Hamiltonian mechanics on a Riemannian manifold plus an Ornstein-Uhlenbeck thermostat.
:::

:::{admonition} The Six Terms in the Momentum Equation
:class: feynman-added note

| Term | Expression | Physical Analogy | Effect |
|------|------------|------------------|--------|
| **Gradient** | $-\nabla\Phi_{\text{eff}}$ | Gravity | Pulls toward low potential |
| **Friction** | $-\gamma p$ | Viscous drag | Damps toward equilibrium |
| **Control** | $u_\pi$ | Rocket thrust | Policy-directed force |
| **Lorentz** | $\beta_{\text{curl}} \mathcal{F} G^{-1} p$ | Magnetic force | Induces rotation/orbiting |
| **Geodesic** | $-\Gamma(G^{-1}p, G^{-1}p)$ | Coriolis/centrifugal | Keeps motion on manifold |
| **Noise** | $\sqrt{2\gamma T_c} G^{1/2} dW$ | Thermal fluctuation | Exploration + equilibrium sampling |

The friction-noise pair implements an **Ornstein-Uhlenbeck thermostat**.
:::

:::{prf:proposition} Explicit Christoffel Symbols for Poincaré Disk
:label: prop-explicit-christoffel-symbols-for-poincare-disk

For the Poincare disk model with metric $G_{ij} = \frac{4\delta_{ij}}{(1-|z|^2)^2}$, the Christoffel symbols in Cartesian coordinates are:

$$
\Gamma^k_{ij}(z) = \frac{2}{1-|z|^2}\left(\delta^k_i z_j + \delta^k_j z_i - \delta_{ij} z^k\right).

$$
The geodesic correction term $\Gamma^k_{ij}\dot{z}^i\dot{z}^j$ contracts to:

$$
\Gamma^k_{ij}\dot{z}^i\dot{z}^j = \frac{4(z \cdot \dot{z})}{1-|z|^2}\dot{z}^k - \frac{2|\dot{z}|^2}{1-|z|^2}z^k.

$$
*Proof.* Direct computation from $\Gamma^k_{ij} = \frac{1}{2}G^{k\ell}(\partial_i G_{j\ell} + \partial_j G_{i\ell} - \partial_\ell G_{ij})$ using $\partial_m[(1-|z|^2)^{-2}] = 4z_m(1-|z|^2)^{-3}$. $\square$

*Geometric interpretation:* The first term $(z \cdot \dot{z})\dot{z}$ accelerates motion radially when moving outward; the second term $|\dot{z}|^2 z$ provides centripetal correction. Together they ensure geodesics are circular arcs perpendicular to the boundary.

:::

:::{div} feynman-prose
Why should you care about the explicit Christoffel symbols? Because they tell you something beautiful about the geometry.

On the Poincare disk, the geodesics---the "straight lines"---are not straight at all in Euclidean terms. They are arcs of circles that hit the boundary at right angles. The Christoffel symbols encode this. When you are moving outward, you need to curve your trajectory to stay on a geodesic. When you are moving tangentially, you need a centripetal correction.

The formula might look complicated, but it is just saying: "adjust your direction so that you curve in the right way for this particular geometry."
:::

:::{prf:definition} Mass Evolution - Jump Process
:label: def-mass-evolution-jump-process

The importance weight $m(s)$ evolves according to a coupled jump-diffusion:

$$
dm = m \cdot r(z, a)\,ds + m \cdot (\eta - 1)\,dN_s,

$$
where:
- $r(z, a)$ is the **reaction rate** from the WFR dynamics ({ref}`Section 20.2 <sec-the-wfr-metric>`)
- $N_s$ is a Poisson process with intensity $\lambda_{\text{jump}}(z)$
- $\eta$ is the multiplicative jump factor (typically $\eta > 1$ for jumps to higher-value charts)

*Interpretation:* Between jumps, mass evolves smoothly via the reaction term $r$. At jump times, the mass is rescaled by factor $\eta$, and the position is teleported via the chart transition operator $L_{i \to j}$.

:::

:::{div} feynman-prose
Now here is the discrete part of the dynamics. The mass $m$ is like a betting stake---it tells you how much probability weight this particular trajectory carries.

Between jumps, the mass can grow or shrink according to the reaction rate $r$. If you are in a good region (positive $r$), your mass grows---you become more confident that this is the right trajectory. If you are in a bad region (negative $r$), your mass shrinks---you become less confident.

At jump times, something more dramatic happens. You teleport to a different chart, and your mass gets rescaled by a factor $\eta$. Typically $\eta > 1$ because you only bother jumping if the new chart is better than the old one. The jump is like a sudden insight: "Wait, I should be thinking about this completely differently!" And when you have that insight, you become more confident.

This is exactly the resampling step in particle filtering. If you are running multiple particles (multiple trajectories in parallel), the ones with high mass survive and the ones with low mass get killed off. Selection pressure, in a mathematical form.
:::

:::{prf:proposition} Jump Intensity from Value Discontinuity
:label: prop-jump-intensity-from-value-discontinuity

The jump intensity $\lambda_{\text{jump}}(z)$ is determined by the value difference across chart boundaries:

$$
\lambda_{\text{jump}}(z) = \lambda_0 \cdot \exp\left(\beta_{\text{ent}} \cdot \left( V_{\text{target}}(L(z)) - V_{\text{source}}(z) - c_{\text{transport}} \right) \right),

$$
where:
- $\lambda_0 > 0$ is a base jump rate
- $\beta_{\text{ent}} > 0$ is the inverse temperature (sharpness)
- $V_{\text{target}}$ and $V_{\text{source}}$ are the value functions on the target and source charts
- $L: \mathcal{Z}_{\text{source}} \to \mathcal{Z}_{\text{target}}$ is the chart transition operator
- $c_{\text{transport}} \ge 0$ is the transport cost (WFR term)

*Remark (SMC Interpretation).* The mass $m(s)$ is precisely the **importance weight** in Sequential Monte Carlo (SMC) / particle filtering. The agent is a single-particle realization of the WFR flow from {ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`. Multiple particles can be used for ensemble-based generation.

**Cross-references:** {ref}`Section 20.2 <sec-the-wfr-metric>` ({prf:ref}`def-the-wfr-action`), {ref}`Section 20.6 <sec-the-unified-world-model>` (WFR world model), {ref}`Section 11 <sec-intrinsic-motivation-maximum-entropy-exploration>` (Filtering and projection).

:::

:::{div} feynman-prose
The jump intensity formula is a Boltzmann factor. The probability of jumping is exponential in the value difference, minus a transport cost. This is exactly the right behavior:

- If the target chart has much higher value, $\lambda$ is large---you jump frequently
- If the target chart has lower value, $\lambda$ is small---why bother?
- The transport cost $c_{\text{transport}}$ provides a barrier: even if the grass looks greener, there is a cost to jumping the fence

The parameter $\beta_{\text{ent}}$ controls how "sharp" this decision is. At high $\beta_{\text{ent}}$, the agent is decisive: if the value difference is positive, jump; if negative, don't. At low $\beta_{\text{ent}}$, the agent is more exploratory: it might jump even to slightly worse charts, just to see what is there.
:::

(sec-the-unified-effective-potential)=
## The Unified Effective Potential

:::{div} feynman-prose
We have been talking about the potential $\Phi$ that the agent rolls down. But where does this potential come from? It is not given to us by Nature; we have to construct it from the quantities we actually care about.

It turns out there are three natural contributions:

1. **The hyperbolic potential** $U$: This drives expansion from the origin toward the boundary. It is the "generation drive"---the urge to create, to produce output, to sample from the model.

2. **The value function** $V_{\text{critic}}$: This is what RL calls the critic. It tells you how good a state is in terms of expected future reward. If you are doing pure control (trying to maximize reward), you roll down this potential.

3. **The risk penalty** $\Psi_{\text{risk}}$: This is caution. Some regions are dangerous---high variance, unstable representations. You want to stay away from them.

The effective potential is a weighted combination of these three. The weight $\alpha$ controls the balance between generation and control. When $\alpha = 1$, you are doing pure generation---expanding outward following the hyperbolic flow. When $\alpha = 0$, you are doing pure control---following the value gradient to maximize reward. In between, you are doing both at once.

This is the key to understanding the agent as a generative model: it is not *either* a generator *or* a controller. It is both, blended together through this unified potential.
:::

The effective potential unifies three terms: the hyperbolic information potential $U$ from holographic generation ({ref}`Section 21.1 <sec-hyperbolic-volume-and-entropic-drift>`), the learned value function $V$ from control ({ref}`Section 2.7 <sec-the-hjb-correspondence>`), and the risk-stress contribution $\Psi_{risk}$ from the stress-energy tensor ({ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`).

:::{prf:definition} Effective Potential
:label: def-effective-potential

The unified effective potential is:

$$
\Phi_{\text{eff}}(z, K) = \alpha\, U(z) + (1 - \alpha)\, V_{\text{critic}}(z, K) + \gamma_{risk}\, \Psi_{\text{risk}}(z),

$$
where:
- $U(z) = -d_{\mathbb{D}}(0, z) = -2\operatorname{artanh}(|z|)$ is the **hyperbolic information potential** (Definition {prf:ref}`def-hyperbolic-information-potential`)
- $V_{\text{critic}}(z, K)$ is the **learned value/critic function** on chart $K$ ({ref}`Section 2.7 <sec-the-hjb-correspondence>`)
- $\Psi_{\text{risk}}(z) = \frac{1}{2}\operatorname{tr}(T_{ij} G^{ij})$ is the **risk-stress contribution** (Theorem {prf:ref}`thm-capacity-constrained-metric-law`)
- $\alpha \in [0, 1]$ is the generation-vs-control hyperparameter
- $\gamma_{risk} \ge 0$ is the risk aversion coefficient

Units: $[\Phi_{\text{eff}}] = \mathrm{nat}$.

:::

:::{admonition} Example: What the Agent "Feels"
:class: feynman-added example

Imagine the agent at position $z$ in the latent space:

- The **hyperbolic term** pulls it outward toward the boundary (generation drive)
- The **value term** pulls it toward high-reward regions (control drive)
- The **risk term** pushes it away from uncertain regions (caution)

The effective potential is the superposition of these three force fields. The agent rolls downhill in this combined landscape.

At $\alpha = 0.5$ (balanced): The agent generates while seeking reward, but avoids risky regions.
:::

:::{prf:proposition} Mode Interpretation
:label: prop-mode-interpretation

The parameter $\alpha$ interpolates between pure generation and pure control:

| Regime              | $\alpha$ Value      | Behavior                                                       |
|---------------------|---------------------|----------------------------------------------------------------|
| **Pure Generation** | $\alpha = 1$        | Flow follows $-\nabla_G U$ (holographic expansion, {ref}`Section 21 <sec-radial-generation-entropic-drift-and-policy-control>`) |
| **Pure Control**    | $\alpha = 0$        | Flow follows $-\nabla_G V_{\text{critic}}$ (policy gradient)   |
| **Hybrid**          | $\alpha \in (0, 1)$ | Balanced generation and control                                |

*Remark (Risk Modulation).* The $\gamma_{risk}$ term provides an additional penalty in high-stress regions (large $T_{ij}$), which further discourages risky trajectories beyond the geometric slowdown from Mass=Metric.

:::
:::{prf:corollary} Gradient Decomposition
:label: cor-gradient-decomposition

The gradient of the effective potential decomposes as:

$$
\nabla_G \Phi_{\text{eff}} = \alpha\, \nabla_G U + (1 - \alpha)\, \nabla_G V_{\text{critic}} + \gamma_{risk}\, \nabla_G \Psi_{\text{risk}}.

$$
For the Poincare disk model, the first term simplifies to:

$$
\nabla_G U = -\frac{(1-|z|^2)}{2}\, \hat{z}, \qquad \hat{z} = \frac{z}{|z|}.

$$
**Cross-references:** Definition {prf:ref}`def-hyperbolic-information-potential`, {ref}`Section 2.7 <sec-the-hjb-correspondence>` (Critic $V$), Section 14.2 (MaxEnt control), Theorem {prf:ref}`thm-capacity-constrained-metric-law`.

*Forward reference (Scalar Field Interpretation).* {ref}`Section 24 <sec-the-reward-field-value-forms-and-hodge-geometry>`
provides the complete field-theoretic interpretation of $V_{\text{critic}}$: the Critic solves the **Screened Poisson
Equation** (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`) with rewards as boundary flux (scalar charges in the
conservative case; Definition {prf:ref}`def-the-reward-flux`), the Value represents **Gibbs Free Energy** (Axiom
{prf:ref}`ax-the-boltzmann-value-law`), and the Value Hessian induces a **Conformal Coupling** to the metric (Definition
{prf:ref}`def-value-metric-conformal-coupling`).

:::

:::{div} feynman-prose
Let me say something about the hyperbolic gradient $\nabla_G U$ on the Poincare disk. The formula is:

$$
\nabla_G U = -\frac{(1-|z|^2)}{2}\, \hat{z}

$$

What does this mean? It means the force points *outward*, toward the boundary, in the radial direction $\hat{z} = z/|z|$. And the magnitude scales like $(1-|z|^2)/2$, which is big near the center and goes to zero at the boundary.

This is exactly right. Near the center, there is a strong drive to expand outward---to generate, to differentiate, to create structure. Near the boundary, this drive weakens, because you are already at high specificity. You have already committed to a particular output.

The factor of $(1-|z|^2)$ is the inverse of the conformal factor. It compensates for the metric blowup at the boundary. In terms of *coordinate* velocity, the force looks like it is weakening. But in terms of *proper* velocity (measured with the metric), the expansion drive stays roughly constant until you get very close to the boundary.
:::

:::{prf:definition} Cognitive Temperature
:label: def-cognitive-temperature

The **cognitive temperature** $T_c > 0$ is the exploration-exploitation tradeoff parameter that controls:

1. **Diffusion magnitude:** The thermal noise term in the geodesic SDE scales as $\sqrt{2T_c}\,dW$
2. **Boltzmann policy:** The softmax temperature in $\pi(a|z) \propto \exp(Q(z,a)/T_c)$
3. **Free energy tradeoff:** The entropy-energy balance $\Phi = E - T_c S$

*Units:* nat (dimensionless in natural units where $k_B = 1$).

*Correspondence:* $T_c$ is the agent-theoretic analogue of thermodynamic temperature $k_B T$ in statistical mechanics.
:::

:::{div} feynman-prose
The cognitive temperature $T_c$ deserves special attention, because it is one of those parameters that looks simple but controls a lot.

At high $T_c$: The agent explores aggressively. The noise term dominates, the policy becomes diffuse, the free energy prefers entropy over energy. The agent is "hot"---it moves around a lot, tries many things, does not commit.

At low $T_c$: The agent exploits what it knows. The noise term is small, the policy becomes sharp, the free energy prefers energy over entropy. The agent is "cold"---it moves decisively toward the best known option.

This is exactly the exploration-exploitation tradeoff, but now it has a thermodynamic interpretation. And it comes with all the machinery of statistical mechanics: the Boltzmann distribution, the free energy, the fluctuation-dissipation theorem.

One beautiful consequence: the agent can *cool down* as it learns. Start with high $T_c$ to explore the space. As you find good regions, lower $T_c$ to commit to them. This is simulated annealing, but it emerges naturally from the thermodynamic structure.
:::

(sec-the-geodesic-baoab-integrator)=
## The Geodesic Boris-BAOAB Integrator

:::{div} feynman-prose
Now we get to the practical question: how do you actually simulate this? You have a stochastic differential equation on a curved manifold with multiple force terms. You cannot just use Euler's method.

The key insight is **operator splitting**. Instead of trying to handle everything at once, you split the dynamics into pieces and handle each piece separately. This is the BAOAB scheme:

- **B** for "kick" (the potential gradient)
- **A** for "drift" (moving along the manifold)
- **O** for "Ornstein-Uhlenbeck" (the thermostat, handling the noise)

The name BAOAB tells you the order: half-kick, half-drift, full thermostat, half-drift, half-kick. This symmetric structure is important---it gives you better accuracy and better preservation of the equilibrium distribution.

Why add Boris to BAOAB? Because we have the Lorentz force, which depends on velocity. The standard kick step does not handle velocity-dependent forces correctly. The Boris algorithm is a trick from plasma physics: instead of applying the force directly, you rotate the momentum around the force axis. This preserves the norm of the momentum, which is exactly what the Lorentz force should do (it does no work, just changes direction).

The combination---Boris-BAOAB---handles everything: the potential gradient, the Lorentz force, the geodesic correction, and the thermal noise. It is a bit complicated, but it is the right way to do it.
:::

We provide the numerical integrator for the controlled geodesic SDE (Definition {prf:ref}`def-bulk-drift-continuous-flow`). The **Boris-BAOAB** scheme extends the standard BAOAB {cite}`leimkuhler2016computation` to handle the velocity-dependent Lorentz force from non-conservative reward fields.

:::{prf:definition} Boris-BAOAB Splitting
:label: def-baoab-splitting

The Boris-BAOAB integrator splits the Lorentz-Langevin dynamics into five substeps per time step $h$:

1. **B** (half kick + Boris rotation):
   - Half-kick from gradient: $p^- \leftarrow p - \frac{h}{2}\nabla\Phi(z)$
   - Boris rotation (if $\mathcal{F} \neq 0$):
     - $t \leftarrow \frac{h}{2}\beta_{\text{curl}} G^{-1}\mathcal{F}$ (rotation vector)
     - $p' \leftarrow p^- + p^- \times t$
     - $s \leftarrow \frac{2t}{1 + |t|^2}$
     - $p^+ \leftarrow p^- + p' \times s$
   - Half-kick from gradient: $p \leftarrow p^+ - \frac{h}{2}\nabla\Phi(z)$

2. **A** (half drift): $z \leftarrow \operatorname{Exp}_z\left(\frac{h}{2} G^{-1}(z)\, p\right)$

3. **O** (thermostat): $p \leftarrow c_1 p + c_2\, G^{1/2}(z)\, \xi$, where $\xi \sim \mathcal{N}(0, I)$

4. **A** (half drift): $z \leftarrow \operatorname{Exp}_z\left(\frac{h}{2} G^{-1}(z)\, p\right)$

5. **B** (half kick + Boris rotation): Same as step 1

where $c_1 = e^{-\gamma h}$ and $c_2 = \sqrt{(1 - c_1^2) T_c}$.

**Conservative Limit:** When $\mathcal{F} = 0$, the Boris rotation is identity and we recover standard BAOAB.

*Remark (Boris Rotation).* The Boris algorithm is a volume-preserving integrator for magnetic-like forces. It rotates the momentum around the local Value Curl axis, preserving the norm $|p|$ while changing direction. This ensures the Lorentz force does no net work, consistent with physics.

*Remark (O-step).* The O-step implements the **Ornstein-Uhlenbeck thermostat**, which exactly preserves the Maxwell-Boltzmann momentum distribution $p \sim \mathcal{N}(0, T_c G)$.

:::

:::{admonition} Why Splitting Works
:class: feynman-added tip

Each sub-step has an exact solution:
- **B-step** (kick): $p \to p - \frac{h}{2}\nabla\Phi$ is just adding a constant to momentum
- **A-step** (drift): $z \to \exp_z(\frac{h}{2}v)$ is following a geodesic
- **O-step** (thermostat): $p \to c_1 p + c_2 \xi$ is an Ornstein-Uhlenbeck step

None of these is approximate---each sub-step is exact. The only approximation is in the splitting itself: we pretend the forces are constant during each sub-step. This is why symmetric splitting (B-A-O-A-B) is important: the errors from the first half cancel the errors from the second half, giving you $O(h^2)$ accuracy instead of $O(h)$.
:::

**Algorithm 22.4.2 (Full Geodesic BAOAB with Jump Step).**

```python
import torch
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Callable


@dataclass
class GeodesicState:
    """State of the geodesic integrator."""
    z: torch.Tensor          # [B, d] latent position
    p: torch.Tensor          # [B, d] momentum (covariant)
    K: torch.Tensor          # [B] chart index (integer)
    m: torch.Tensor          # [B] importance weight (mass)
    s: float                 # computation time


def poincare_metric(z: torch.Tensor) -> torch.Tensor:
    """
    Poincare disk metric tensor G_{ij}(z).

    G_{ij} = 4 delta_{ij} / (1 - |z|^2)^2

    Returns: [B, d, d] metric tensor
    """
    B, d = z.shape
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
    conformal_factor = 4.0 / (1.0 - r_sq + 1e-8) ** 2  # [B, 1]
    return conformal_factor.unsqueeze(-1) * torch.eye(d, device=z.device).expand(B, d, d)


def poincare_metric_inv(z: torch.Tensor) -> torch.Tensor:
    """
    Inverse Poincare metric G^{ij}(z).

    G^{ij} = (1 - |z|^2)^2 / 4 * delta^{ij}
    """
    B, d = z.shape
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    inv_conformal = (1.0 - r_sq + 1e-8) ** 2 / 4.0
    return inv_conformal.unsqueeze(-1) * torch.eye(d, device=z.device).expand(B, d, d)


def mobius_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Möbius addition in the Poincare disk: a ⊕ b.

    (a + b) / (1 + <a, b>)  [simplified for small b]

    Full formula:
    a ⊕ b = ((1 + 2<a,b> + |b|^2) a + (1 - |a|^2) b) / (1 + 2<a,b> + |a|^2|b|^2)
    """
    a_sq = (a ** 2).sum(dim=-1, keepdim=True)
    b_sq = (b ** 2).sum(dim=-1, keepdim=True)
    a_dot_b = (a * b).sum(dim=-1, keepdim=True)

    num = (1 + 2*a_dot_b + b_sq) * a + (1 - a_sq) * b
    denom = 1 + 2*a_dot_b + a_sq * b_sq + 1e-8

    return num / denom


def poincare_exp_map(z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Exponential map on Poincare disk: Exp_z(v).

    Exp_z(v) = φ_{-z}(tanh(||v||_z / 2) * v / ||v||)

    where φ_{-z} is Möbius translation and ||v||_z is the hyperbolic norm.
    """
    # Compute hyperbolic norm of v at z
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    lambda_z = 2.0 / (1.0 - r_sq + 1e-8)  # conformal factor
    v_norm = torch.sqrt((v ** 2).sum(dim=-1, keepdim=True) + 1e-8) * lambda_z

    # Direction (normalized in Euclidean sense)
    v_dir = v / (torch.sqrt((v ** 2).sum(dim=-1, keepdim=True)) + 1e-8)

    # Magnitude in disk: tanh(||v||_z / 2)
    magnitude = torch.tanh(v_norm / 2.0)

    # Point at origin in direction v_dir with magnitude
    w = magnitude * v_dir

    # Translate by -z (Möbius addition)
    return mobius_add(z, w)


def christoffel_contraction(z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute Γ^k_{ij} v^i v^j for Poincare disk.

    For conformal metric G = λ²I with λ = 2/(1-|z|²), the Christoffel contraction is:
    Γ^k_{ij} v^i v^j = 4(z·v)v^k/(1-|z|²) - 2|v|²z^k/(1-|z|²)

    Derivation: For G_{ij} = e^{2φ}δ_{ij}, Γ^k_{ij}v^iv^j = 2(v·∇φ)v^k - |v|²∂^kφ
    With φ = log(2/(1-|z|²)), we have ∂^kφ = 2z^k/(1-|z|²).
    """
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    v_sq = (v ** 2).sum(dim=-1, keepdim=True)
    z_dot_v = (z * v).sum(dim=-1, keepdim=True)
    one_minus_r_sq = 1.0 - r_sq + 1e-8

    # Γ^k v^i v^j = 4(z·v)v^k/(1-|z|²) - 2|v|²z^k/(1-|z|²)
    term1 = (4.0 / one_minus_r_sq) * z_dot_v * v
    term2 = -(2.0 / one_minus_r_sq) * v_sq * z

    return term1 + term2


def geodesic_baoab_step(
    state: GeodesicState,
    grad_Phi: torch.Tensor,           # [B, d] gradient of effective potential
    u_pi: torch.Tensor,               # [B, d] control field from policy
    T_c: float,                       # cognitive temperature
    gamma: float,                     # friction coefficient
    h: float,                         # time step
    jump_rate_fn: Optional[Callable] = None,  # λ(z, K) -> [B]
    chart_transition_fn: Optional[Callable] = None,  # L(z, K_src, K_tgt) -> z'
    value_fn: Optional[Callable] = None,  # V(z, K) -> [B]
) -> GeodesicState:
    """
    Full Geodesic BAOAB integrator with Poisson jump process.

    Implements Algorithm 22.4.2:
    1. B-step: half kick from potential + control
    2. A-step: half drift via exponential map
    3. O-step: Ornstein-Uhlenbeck thermostat
    4. A-step: half drift
    5. B-step: half kick
    6. Jump-step: Poisson process for chart transitions

    Cross-references:
        - Definition 22.2.1 (Bulk Drift SDE)
        - Definition 22.2.2 (Jump Process)
        - {ref}`Section 2.5.1 <sec-levi-civita-connection-and-parallel-transport>` (Christoffel symbols)
    """
    z, p, K, m = state.z, state.p, state.K, state.m
    B, d = z.shape
    device = z.device

    # BAOAB coefficients
    c1 = math.exp(-gamma * h)
    c2 = math.sqrt((1 - c1**2) * T_c) if T_c > 0 else 0.0

    # ===== B-step: half kick =====
    # p ← p - (h/2) * (∇Φ_eff - u_π)
    total_force = grad_Phi - u_pi  # Note: gradient is positive, so subtract
    p = p - (h / 2) * total_force

    # ===== A-step: half drift =====
    # z ← Exp_z((h/2) G^{-1} p)
    G_inv = poincare_metric_inv(z)
    velocity = torch.einsum('bij,bj->bi', G_inv, p)  # contravariant velocity

    # Apply geodesic correction to velocity
    geodesic_corr = christoffel_contraction(z, velocity)
    velocity_corrected = velocity - (h / 4) * geodesic_corr  # half of half-step

    z = poincare_exp_map(z, (h / 2) * velocity_corrected)

    # ===== O-step: thermostat =====
    # p ← c₁ p + c₂ G^{1/2} ξ
    G = poincare_metric(z)
    # G^{1/2} via Cholesky (for diagonal, just sqrt of diagonal)
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    G_sqrt_factor = 2.0 / (1.0 - r_sq + 1e-8)  # sqrt of conformal factor

    xi = torch.randn_like(p)
    p = c1 * p + c2 * G_sqrt_factor * xi

    # ===== A-step: half drift =====
    G_inv = poincare_metric_inv(z)
    velocity = torch.einsum('bij,bj->bi', G_inv, p)
    geodesic_corr = christoffel_contraction(z, velocity)
    velocity_corrected = velocity - (h / 4) * geodesic_corr

    z = poincare_exp_map(z, (h / 2) * velocity_corrected)

    # ===== B-step: half kick =====
    # Recompute gradient at new position (for accuracy)
    # In practice, often reuse grad_Phi for efficiency
    p = p - (h / 2) * total_force

    # ===== Jump-step: Poisson process =====
    if jump_rate_fn is not None and chart_transition_fn is not None:
        # Compute jump probability
        lambda_jump = jump_rate_fn(z, K)  # [B]
        prob_jump = 1 - torch.exp(-lambda_jump * h)

        # Sample jumps
        u = torch.rand(B, device=device)
        jumps = u < prob_jump  # [B] boolean

        if jumps.any() and value_fn is not None:
            # Determine target chart (simplified: assume single target)
            K_target = (K + 1) % 4  # Example: cycle through 4 charts

            # Apply chart transition for jumping particles
            z_new = chart_transition_fn(z, K, K_target)
            z = torch.where(jumps.unsqueeze(-1), z_new, z)
            K = torch.where(jumps, K_target, K)

            # Update mass (importance weight)
            eta = 1.1  # Jump mass factor
            m = torch.where(jumps, m * eta, m)

    # Project to ensure we stay in disk
    z_norm = torch.sqrt((z ** 2).sum(dim=-1, keepdim=True))
    z = torch.where(z_norm > 0.999, z * 0.999 / z_norm, z)

    return GeodesicState(z=z, p=p, K=K, m=m, s=state.s + h)
```

:::
:::{prf:proposition} BAOAB Preserves Boltzmann
:label: prop-baoab-preserves-boltzmann

The BAOAB integrator preserves the Boltzmann distribution $\rho(z, p) \propto \exp(-\Phi_{\text{eff}}(z)/T_c - \|p\|_G^2 / (2T_c))$ to second order in $h$.

*Proof sketch.* The symmetric splitting B-A-O-A-B ensures time-reversibility of the deterministic steps. The O-step exactly samples the Maxwell-Boltzmann momentum distribution. Together, these guarantee that $\rho$ is a fixed point of the numerical flow up to $O(h^3)$ errors. See {cite}`leimkuhler2016computation`. $\square$

*Remark (Comparison to Euler-Maruyama).* Euler-Maruyama has $O(h)$ bias in the stationary distribution, whereas BAOAB achieves $O(h^2)$. For long trajectories, this difference is critical.

:::

:::{div} feynman-prose
This result about preserving the Boltzmann distribution is crucial. When you run a simulation, you want it to sample from the correct distribution. If your integrator has a bias, your samples will be wrong---you will be over-sampling some regions and under-sampling others.

Euler-Maruyama, the simplest integrator, has $O(h)$ bias. That means if you use a time step of $h = 0.01$, your distribution is wrong by about 1%. Sounds small? It adds up. Over a million steps, the errors compound.

BAOAB has $O(h^2)$ bias. Same time step, $h = 0.01$, the error is about 0.01%. A hundred times smaller. This matters enormously for any application where you need accurate statistics---which is basically all of them.

The price you pay is five sub-steps instead of one. But that is a small price for a hundred-fold improvement in accuracy.
:::

(pi-langevin-thermostat)=
::::{admonition} Physics Isomorphism: Langevin Thermostat
:class: note

**In Physics:** The Langevin equation $m\ddot{x} = -\nabla U - \gamma\dot{x} + \sqrt{2\gamma k_B T}\,\xi(t)$ describes Brownian motion in a potential with friction $\gamma$ and thermal noise. The Ornstein-Uhlenbeck thermostat samples the Maxwell-Boltzmann distribution $p \propto \exp(-mv^2/2k_BT)$ {cite}`leimkuhler2016computation`.

**In Implementation:** The BAOAB integrator (Definition {prf:ref}`def-baoab-splitting`) splits the dynamics:
- **B:** $p \gets p - \frac{h}{2}\nabla_z\Phi_{\text{eff}}$ (kick)
- **A:** $z \gets z + \frac{h}{2}G^{-1}p$ (drift)
- **O:** $p \gets c_1 p + c_2 G^{1/2}\xi$ (thermostat)
- **A, B:** repeat

**Correspondence Table:**
| Molecular Dynamics | Agent (BAOAB) |
|:-------------------|:--------------|
| Position $x$ | Latent state $z$ |
| Momentum $p$ | Auxiliary variable $p$ |
| Potential $U(x)$ | Effective potential $\Phi_{\text{eff}}(z)$ |
| Friction $\gamma$ | Damping coefficient |
| Temperature $k_B T$ | Cognitive temperature $T_c$ |
| Maxwell-Boltzmann | Stationary policy distribution |

**Advantage:** BAOAB preserves the Boltzmann distribution to $O(h^2)$ (Proposition {prf:ref}`prop-baoab-preserves-boltzmann`), avoiding the $O(h)$ bias of Euler-Maruyama.
::::

(pi-detailed-balance)=
::::{admonition} Physics Isomorphism: Detailed Balance
:class: note

**In Physics:** A stochastic process satisfies detailed balance if transition rates satisfy $\pi(x)W(x \to y) = \pi(y)W(y \to x)$ for all states $x, y$. This implies the stationary distribution $\pi$ and time-reversibility {cite}`vanKampen1992stochastic`.

**In Implementation:** The WFR dynamics satisfy detailed balance at equilibrium:

$$
\rho_*(z) \cdot J(z \to z') = \rho_*(z') \cdot J(z' \to z)

$$
where $\rho_* \propto \exp(-\Phi_{\text{eff}}/T_c)\sqrt{|G|}$ is the Boltzmann distribution.

**Correspondence Table:**
| Statistical Mechanics | Agent (Equilibrium) |
|:----------------------|:--------------------|
| Transition rate $W(x \to y)$ | Jump rate $\lambda_{KK'}$ |
| Stationary distribution $\pi$ | Equilibrium belief $\rho_*$ |
| Detailed balance | Reversibility at Nash |
| Entropy production $\dot{S}$ | Zero at equilibrium |
| Fluctuation-dissipation | Einstein relation for $T_c$ |

**Consequence (conservative case):** Detailed balance ensures the BAOAB thermostat samples the correct distribution. When $\mathcal{F} \neq 0$, detailed balance is broken and the steady state is a NESS.
::::

(sec-the-overdamped-limit)=
## The Overdamped Limit

:::{div} feynman-prose
So far we have been working with the full second-order dynamics: position *and* momentum. But in many applications, you do not need all that machinery. When friction is strong enough, the momentum equilibrates almost instantly to the force, and you can forget about it.

This is the **overdamped limit**, and it is important for two reasons:

1. **Simplicity**: First-order dynamics are easier to simulate and analyze than second-order
2. **Relevance**: Many real systems operate in this regime---diffusion models, Brownian motion in viscous fluids, biological processes

The mathematical statement is: when friction $\gamma$ is large compared to the forces, the velocity quickly relaxes to

$$
\dot{z} \approx \mathcal{M}_\gamma\!\left(-G^{-1}\nabla\Phi\right), \qquad \mathcal{M}_\gamma := (\gamma I - \beta_{\text{curl}} G^{-1}\mathcal{F})^{-1}.
$$
You do not need to track the momentum explicitly; you can just compute it from the force using the curl-corrected mobility.

The resulting dynamics are curl-corrected gradient flow with noise: roll downhill, get jostled by thermal fluctuations, and (if $\mathcal{F} \neq 0$) pick up sideways drift through $\mathcal{M}_\gamma$. No inertia, no coasting, no overshoot. The agent responds instantaneously to changes in the potential.

When is this a good approximation? When the timescale of momentum relaxation ($\sim 1/\gamma$) is much shorter than the timescale of position changes. In that case, the momentum "slaves" to the position, and you can eliminate it.
:::

In many applications (diffusion models, biological control), the system operates in the **overdamped regime** where friction dominates inertia. We derive this limit rigorously.

:::{prf:theorem} Overdamped Limit
:label: thm-overdamped-limit

Consider the second-order SDE from Definition {prf:ref}`def-bulk-drift-continuous-flow` with friction coefficient $\gamma$:

$$
m\,\ddot{z}^k + \gamma\,\dot{z}^k - \beta_{\text{curl}} G^{km}\mathcal{F}_{mj}\dot{z}^j + G^{kj}\partial_j\Phi + \Gamma^k_{ij}\dot{z}^i\dot{z}^j = \sqrt{2T_c}\,\left(G^{-1/2}\right)^{kj}\,\xi^j,

$$
where $m$ is the "inertial mass" and $\xi$ is white noise. In the limit $\gamma \to \infty$ with $m$ fixed (or equivalently, $m \to 0$ with $\gamma$ fixed), the dynamics reduce to the first-order Langevin equation:

$$
dz^k = \left[\mathcal{M}_\gamma(z)\right]^{k}{}_{j}\left(-G^{j\ell}(z)\,\partial_\ell\Phi_{\text{gen}}(z)\right) ds + \sqrt{2T_c}\,\left(G^{-1/2}(z)\right)^{kj}\,dW^j_s.

$$
*Proof sketch.* In the high-friction limit, velocity equilibrates instantaneously to
$\dot{z} \approx \mathcal{M}_\gamma(-G^{-1}\nabla\Phi)$. The geodesic term
$\Gamma(\dot{z},\dot{z}) \sim O(|\dot{z}|^2) = O(\gamma^{-2})$ is negligible. What remains is the curl-corrected
gradient flow with diffusion. See {ref}`Appendix A.4 <sec-appendix-a-full-derivations>` for the full singular
perturbation analysis. $\square$

:::

:::{div} feynman-prose
Notice what disappears in the overdamped limit: the geodesic correction term $\Gamma(\dot{z},\dot{z})$. This makes sense. The geodesic correction is about inertia---it tells you how to maintain your direction on a curved surface when you are coasting. But in the overdamped limit, you are not coasting. You are always being dragged to a halt by friction and then pushed by the force. There is no inertia to correct for.

This is why the overdamped equation is so much simpler: just
$dz = \mathcal{M}_{\text{curl}}\!\left(-G^{-1}\nabla\Phi\right) ds + \text{noise}$ with $\mathcal{M}_{\text{curl}} := (I - \beta_{\text{curl}} G^{-1}\mathcal{F})^{-1}$. The curvature still matters---it is hiding in the metric $G$---but the explicit Christoffel symbols are gone.
:::

:::{prf:corollary} Recovery of Holographic Flow
:label: cor-recovery-of-holographic-flow

Setting $\alpha = 1$ (pure generation), $T_c \to 0$ (deterministic limit), and $\mathcal{F} = 0$ (conservative case) in the overdamped equation recovers the holographic gradient flow from {ref}`Section 21.2 <sec-policy-control-field>`:

$$
\dot{z} = -G^{-1}(z)\,\nabla U(z).

$$
For the Poincare disk, this gives $\dot{z} = \frac{(1-|z|^2)}{2}\,z$, which integrates to $|z(\tau)| = \tanh(\tau/2)$.

*Proof.* Direct substitution of $\Phi_{\text{gen}} = U$ into the overdamped equation. The explicit solution for the radial coordinate $r(\tau) = |z(\tau)|$ satisfies $\dot{r} = \frac{1-r^2}{2}$, which integrates to $r(\tau) = \tanh(\tau/2 + \operatorname{artanh}(r_0))$. For $r_0 = 0$, we get $r(\tau) = \tanh(\tau/2)$. $\square$

*Remark.* This proves that the "ad-hoc" holographic law from {ref}`Section 21 <sec-radial-generation-entropic-drift-and-policy-control>` is actually the **optimal control trajectory** for the geometry defined in {ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`, vindicating the intuition.

:::

:::{div} feynman-prose
This is one of those beautiful moments where everything fits together. In Section 21, we introduced the holographic flow $|z(\tau)| = \tanh(\tau/2)$ as a kind of "natural" radial expansion. It looked like an ad-hoc choice.

But now we see it is not ad-hoc at all. It is the *unique* geodesic flow for the hyperbolic potential on the Poincare disk in the overdamped, deterministic limit. The geometry *forces* this flow on us.

This is the kind of consistency check that tells you the theory is on the right track. Different parts, derived independently, turn out to agree. The holographic law is not just one choice among many---it is the geometrically natural choice.
:::

:::{prf:corollary} Fokker-Planck Duality {cite}`risken1996fokkerplanck`
:label: cor-fokker-planck-duality

The stationary distribution of the overdamped SDE is:

$$
p_*(z) \propto \exp\left(-\frac{\Phi_{\text{gen}}(z)}{T_c}\right)\,\sqrt{|G(z)|},

$$
where $|G| = \det(G)$ is the metric determinant. This is the Boltzmann distribution on the curved manifold.

*Proof.* The Fokker-Planck equation for the overdamped dynamics is:

$$
\partial_s p = \nabla_i\left( G^{ij}\left( p\,\partial_j\Phi + T_c\,\partial_j p \right) \right).

$$
Setting $\partial_s p = 0$ and using detailed balance gives $p \propto e^{-\Phi/T_c} \sqrt{|G|}$. The $\sqrt{|G|}$ factor accounts for the Riemannian volume form. $\square$

**Cross-references:** {ref}`Section 21.2 <sec-policy-control-field>` (Langevin dynamics), Theorem {prf:ref}`thm-equivalence-of-entropy-regularized-control-forms-discrete-macro`, {ref}`Section 2.11 <sec-variance-value-duality-and-information-conservation>` (Belief density evolution).

:::

:::{div} feynman-prose
The Fokker-Planck equation tells you how probability density evolves under the SDE. But we care most about the *stationary* distribution---the long-time limit. And look at that beautiful formula:

$$
p_*(z) \propto \exp\left(-\frac{\Phi}{T_c}\right)\,\sqrt{|G|}

$$

This is the Boltzmann distribution on a curved manifold. The $\exp(-\Phi/T_c)$ part is familiar from statistical mechanics: probability is exponentially suppressed in high-potential regions. The $\sqrt{|G|}$ part is the volume correction from the geometry: probability is enhanced in regions where the metric is large, because there is "more space" there in the intrinsic sense.

Together, these give you the correct equilibrium. The agent, running the SDE for a long time, will sample from this distribution. Low potential, high probability. Large metric volume element, high probability. This is the target distribution for MCMC sampling on curved manifolds.
:::

(pi-fokker-planck)=
::::{admonition} Physics Isomorphism: Fokker-Planck Equation
:class: note

**In Physics:** The Fokker-Planck equation describes the time evolution of probability density under drift and diffusion: $\partial_t p = -\nabla \cdot (p\,\mathbf{F}) + D\nabla^2 p$. On a Riemannian manifold with metric $g$, the diffusion term becomes the Laplace-Beltrami operator {cite}`risken1996fokkerplanck`.

**In Implementation:** The belief density $\rho(z,s)$ evolves via (Corollary {prf:ref}`cor-fokker-planck-duality`):

$$
\partial_s p = \nabla_i\left( G^{ij}\left( p\,\partial_j\Phi_{\text{eff}} + T_c\,\partial_j p \right) \right)

$$
with stationary distribution $p_*(z) \propto \exp(-\Phi_{\text{eff}}(z)/T_c)\sqrt{|G(z)|}$.

**Correspondence Table:**
| Statistical Physics | Agent (Belief Dynamics) |
|:--------------------|:------------------------|
| Probability density $p(x,t)$ | Belief density $\rho(z,s)$ |
| Drift force $\mathbf{F}$ | Effective potential gradient $-\nabla\Phi_{\text{eff}}$ |
| Diffusion constant $D$ | Cognitive temperature $T_c$ |
| Laplacian $\nabla^2$ | Laplace-Beltrami $\Delta_G$ |
| Boltzmann equilibrium | WFR stationary distribution |

**Loss Function:** Stein discrepancy $\mathbb{E}[\|\nabla \log p - \nabla \log p_*\|^2_G]$.
::::

(sec-agent-lifecycle-summary)=
## Agent Lifecycle Summary

:::{div} feynman-prose
Now let me put all the pieces together. We have equations for continuous motion, for discrete jumps, for the potential, for the temperature. How do these combine into a coherent picture of what the agent does?

The agent lifecycle has five phases, and they map beautifully onto a physical picture of phase transitions:

1. **Init**: Start at the origin. This is the "gas phase"---maximum entropy, no commitment, all possibilities open.

2. **Kick**: Apply symmetry-breaking control. This is "nucleation"---you have to pick a direction, break the perfect symmetry of the origin.

3. **Bulk**: Geodesic flow with jumps. This is the "liquid phase"---flowing, exploring, but with structure. The trajectory can switch between charts, try different representations.

4. **Boundary**: Reach the cutoff radius. This is "crystallization"---committing to a specific output, sampling the texture.

5. **Decode**: Map to the output space. The latent trajectory becomes an actual observable.

The beautiful thing is that these phases are not imposed from outside. They *emerge* from the dynamics. The symmetry at the origin creates the need for a kick. The expansion drive creates the bulk flow. The boundary condition creates the stopping criterion. The geometry itself choreographs the whole dance.
:::

The complete agent lifecycle integrates the components from Sections 21-22 into a coherent execution flow.

:::{prf:definition} Agent Lifecycle Phases
:label: def-agent-lifecycle-phases


| Phase           | Time Interval                | Dynamics                         | Texture      | Key Operations                                                                         |
|-----------------|------------------------------|----------------------------------|--------------|----------------------------------------------------------------------------------------|
| **1. Init**     | $\tau = 0$                   | $z(0) = 0$                       | None         | Initialize at origin; $p(0) \sim \mathcal{N}(0, T_c G(0))$                             |
| **2. Kick**     | $[0, \tau_{kick}]$           | Langevin at origin               | None         | Apply symmetry-breaking control $u_\pi$ (Def. {prf:ref}`def-the-control-field`)        |
| **3. Bulk**     | $[\tau_{kick}, \tau_{stop}]$ | BAOAB + Jumps                    | **Firewall** | Geodesic flow with chart transitions                                                   |
| **4. Boundary** | $\tau = \tau_{stop}$         | $\lVert z\rVert \geq R_{cutoff}$ | Sampled      | Sample texture $z_{tex} \sim \mathcal{N}(0, \Sigma(z))$                                |
| **5. Decode**   | Post-$\tau_{stop}$           | —                                | Used         | $x = \text{Decoder}(e_K, z_n, z_{tex})$                                                |

*Remark.* The **Texture Firewall** (Axiom {prf:ref}`ax-bulk-boundary-decoupling`) ensures that $\partial_{z_{tex}} \dot{z} = 0$ throughout the bulk phase—texture is completely invisible to the dynamics.

**Algorithm 22.6.2 (Full Agent Loop).**

```python
def run_agent_loop(
    policy: Policy,
    decoder: Decoder,
    T_c: float,
    gamma: float,
    h: float,
    R_cutoff: float = 0.95,
    max_steps: int = 1000,
) -> torch.Tensor:
    """
    Execute the full agent lifecycle from init to decode.

    Returns: Generated output x
    """
    B, d = 1, policy.latent_dim
    device = policy.device

    # ===== Phase 1: Init =====
    z = torch.zeros(B, d, device=device)
    p = torch.randn(B, d, device=device) * math.sqrt(T_c * 4.0)  # G(0) = 4I
    K = torch.zeros(B, dtype=torch.long, device=device)
    m = torch.ones(B, device=device)
    state = GeodesicState(z=z, p=p, K=K, m=m, s=0.0)

    # ===== Phase 2: Kick =====
    # Apply symmetry-breaking control at origin
    u_pi = policy.symmetry_breaking_kick(z, mode='generation')

    # ===== Phase 3: Bulk (with Texture Firewall) =====
    for step in range(max_steps):
        # Compute effective potential gradient
        grad_Phi = compute_effective_potential_gradient(
            state.z, state.K, policy.value_fn, alpha=0.5
        )

        # Update control field
        u_pi = policy.control_field(state.z, state.K)

        # BAOAB step (texture is invisible here)
        state = geodesic_baoab_step(
            state, grad_Phi, u_pi, T_c, gamma, h,
            jump_rate_fn=policy.jump_rate,
            chart_transition_fn=policy.chart_transition
        )

        # Check boundary condition
        z_norm = torch.sqrt((state.z ** 2).sum(dim=-1))
        if (z_norm >= R_cutoff).all():
            break

    # ===== Phase 4: Boundary - Sample texture =====
    z_tex = sample_holographic_texture(state.z, sigma_tex=0.1)

    # ===== Phase 5: Decode =====
    embedding = policy.chart_embedding(state.K)  # e_K
    x = decoder(embedding, state.z, z_tex)

    return x
```

:::

:::{admonition} The Texture Firewall
:class: feynman-added warning

Notice that during the Bulk phase, the texture is completely invisible. The dynamics depend only on $(z, K, m)$, not on any fine-grained texture information.

This is the **Texture Firewall**: texture is sampled only at the boundary, not during the bulk flow. Why? Because if texture influenced the dynamics, the bulk would become infinitely complex---you would need to track an infinite number of degrees of freedom.

The firewall ensures the bulk dynamics remain finite-dimensional, with all the high-dimensional structure appearing only at the final step.
:::

:::{prf:proposition} Phase Transition Interpretation
:label: prop-phase-transition-interpretation

The agent lifecycle corresponds to a thermodynamic phase transition:

| Phase | Thermodynamic Analogy | Order Parameter |
|-------|----------------------|-----------------|
| Init (gas) | High entropy, symmetric | $\lVert z\rVert = 0$ |
| Kick (nucleation) | Symmetry breaking | $u_\pi \neq 0$ |
| Bulk (liquid) | Directed flow | $0 < \lVert z\rVert < R_{cutoff}$ |
| Boundary (solid) | Crystallization | $\lVert z\rVert \geq R_{cutoff}$ |

:::

(sec-adaptive-thermodynamics)=
## Adaptive Thermodynamics (Fluctuation-Dissipation)

:::{div} feynman-prose
Up to now, we have been treating the temperature $T_c$ as a constant. But there is something unsatisfying about that. The metric $G$ varies across the manifold---should not the temperature vary too?

The answer is yes, if you want to maintain the correct relationship between noise and dissipation. This is the **fluctuation-dissipation theorem**: in equilibrium, the noise and the friction are related by the temperature. If you change one, you have to change the others to stay in equilibrium.

On a curved manifold, this becomes the **Einstein relation**: $\sigma^2 = 2\gamma T_c / G$. The noise variance, the friction, the temperature, and the metric are all tied together.

What happens if you let the temperature adapt to the geometry? Something wonderful: the agent automatically transitions between exploration and exploitation based on where it is. Near the origin (small $G$), the effective noise is large---exploration. Near the boundary (large $G$), the effective noise is small---exploitation. No temperature schedule needed. The geometry does it for you.
:::

The temperature $T_c$ and friction $\gamma$ need not be constant---they can adapt to the local geometry to maintain the Einstein relation.

:::{prf:definition} Einstein Relation on Manifolds
:label: def-einstein-relation-on-manifolds

The fluctuation-dissipation relation requires:

$$
\sigma^2(z) = \frac{2\gamma(z)\, T_c}{G(z)},

$$
where $\sigma^2$ is the noise variance. This ensures the correct equilibrium distribution.

:::
:::{prf:proposition} Automatic Phase Transitions
:label: prop-automatic-phase-transitions

With adaptive temperature $T_c(z)$ satisfying the Einstein relation:

| Regime                      | Metric $G(z)$ | Effective Noise | Phase Behavior                |
|-----------------------------|---------------|-----------------|-------------------------------|
| **Uncertain** (near origin) | Small         | Large           | Gas phase (exploration)       |
| **Certain** (near boundary) | Large         | Small           | Solid phase (crystallization) |

*Remark.* This automatic phase transition emerges from the geometry alone---no explicit temperature schedule is needed.

:::

:::{div} feynman-prose
This is worth pausing on. In standard machine learning, if you want to transition from exploration to exploitation, you have to design a temperature schedule. High temperature at the start, gradually lowering it. This is simulated annealing, and it requires hand-tuning.

Here, the same effect emerges automatically. The geometry provides a "natural" temperature schedule: high effective noise near the center, low effective noise near the boundary. You do not have to tune it; it falls out of the mathematics.

This is another instance of the general principle: **the geometry does the work**. Instead of adding explicit mechanisms for various behaviors, you build the right geometry and let the behaviors emerge.
:::

:::{prf:definition} Fisher-Covariance Duality
:label: def-fisher-covariance-duality

The inverse relationship between uncertainty and metric:

$$
G(z) \approx \Sigma^{-1}(z),

$$
where $\Sigma(z)$ is the posterior covariance of the belief at $z$. This duality underlies the Mass=Metric principle (Definition {prf:ref}`def-mass-tensor`).

**Algorithm 22.7.4 (Adaptive Temperature).**

```python
def adaptive_temperature(
    z: torch.Tensor,
    base_T: float,
    certainty_scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute adaptive temperature based on local geometry.

    T_c(z) = base_T * (1 - |z|^2)^2 / 4

    This maintains constant effective noise: sigma^2 * G = 2 * gamma * T_c
    """
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    # Conformal factor inverse: G^{-1} = (1-|z|^2)^2 / 4
    inv_conformal = (1.0 - r_sq + 1e-8) ** 2 / 4.0
    return base_T * inv_conformal * certainty_scale
```

:::

:::{div} feynman-prose
The Fisher-Covariance duality is one of those deep facts that keeps showing up in different guises. Here is the intuition:

- The **Fisher information** tells you how much information the data provides about the parameters. High Fisher information means the data is very informative.
- The **posterior covariance** tells you how uncertain you are about the parameters after seeing the data. High covariance means high uncertainty.

These are inversely related: $G \approx \Sigma^{-1}$. If the data is very informative (high $G$), you are very certain (low $\Sigma$). If the data is not informative (low $G$), you remain uncertain (high $\Sigma$).

This is why Mass = Metric makes sense. The metric *is* the Fisher information. Where you are certain (high Fisher, high metric), you should be cautious---you have committed to a representation, do not throw it away lightly. Where you are uncertain (low Fisher, low metric), you should be exploratory---you do not have much to lose.
:::

:::{prf:corollary} Deterministic Boundary
:label: cor-deterministic-boundary

As $|z| \to 1$:

$$
T_c(z) \to 0, \qquad \text{noise} \to 0.

$$
The agent becomes deterministic at the boundary, ensuring reproducible outputs.

:::

:::{div} feynman-prose
This corollary is the final piece of the puzzle. At the boundary, the noise goes to zero. The agent becomes deterministic.

Why is this important? Because outputs need to be reproducible. If you generate an image, you want the same latent code to give the same image every time. If there were noise at the boundary, you would get different images each time---maybe good for "creative variability," but terrible for debugging and control.

The geometry gives you both: exploration in the bulk (where noise is large), and determinism at the boundary (where noise vanishes). The stochastic and deterministic regimes are unified in a single framework, with the transition happening smoothly as you approach the boundary.
:::

(sec-summary-tables-and-diagnostic-nodes)=
## Summary Tables and Diagnostic Nodes

:::{div} feynman-prose
Let me summarize what we have built. The equations of motion combine three ingredients:

1. **Geodesic dynamics** on a curved manifold, with the metric providing natural inertia
2. **Stochastic fluctuations** controlled by the cognitive temperature
3. **Discrete jumps** between charts, enabling topological transitions

The key insight is that these are not independent mechanisms. They all emerge from a single variational principle: minimize the Onsager-Machlup action. The geometry (metric, Christoffel symbols, curvature) comes from the capacity constraint. The noise comes from the entropy regularization. The jumps come from the multi-chart structure.

The diagnostic nodes below help you check that everything is working correctly. If the GeodesicCheck is high, your trajectory is not following the controlled geodesic---maybe a bug in the Christoffel computation. If the JumpConsistencyCheck is high, your jump rates are violating detailed balance---maybe the value functions are inconsistent across charts.

These are the kinds of things you want to monitor in a running system. Not just "is the loss going down," but "are the geometric invariants being preserved."
:::

**Summary of Equations of Motion:**

| Equation                 | Expression                                                                                                                  | Regime       | Units                |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------|--------------|----------------------|
| Extended Onsager-Machlup | $S_{\mathrm{OM}} = \int (\frac{1}{2}\mathbf{M}\lVert\dot{z}\rVert^2 + \Phi_{\text{eff}} + \frac{T_c}{12}R + T_c H_\pi)\,ds$ | Path-space   | nat                  |
| Full Geodesic SDE        | $dz = (-G^{-1}\nabla\Phi_{\text{eff}} + u_\pi + \beta_{\text{curl}} G^{-1}\mathcal{F}\dot{z} - \Gamma(\dot{z},\dot{z}))\,ds + \sqrt{2T_c}\,G^{-1/2}\,dW_s$                 | Second-order | $[z]$                |
| Overdamped               | $dz = \mathcal{M}_{\text{curl}}\!\left(-G^{-1}\nabla\Phi_{\text{eff}} + u_\pi\right)\,ds + \sqrt{2T_c}\,G^{-1/2}\,dW_s$                                                      | First-order  | $[z]$                |
| Jump Intensity           | $\lambda_{K\to j} = \lambda_0 \exp(\beta_{\text{ent}}\,\Delta V)$                                                           | Discrete     | step$^{-1}$          |
| Mass = Metric            | $\mathbf{M}(z) \equiv G(z)$                                                                                                 | Kinematic    | $[z]^{-2}$           |
| Texture Covariance       | $\Sigma_{\text{tex}}(z) = \sigma_{\text{tex}}^2\, G^{-1}(z)$                                                                | Boundary     | $[z_{\text{tex}}]^2$ |

**Effective Potential Decomposition:**

$$
\Phi_{\text{eff}}(z, K) = \alpha\,U(z) + (1-\alpha)\,V_{\text{critic}}(z, K) + \gamma_{\text{risk}}\,\Psi_{\text{risk}}(z)

$$
where $\alpha \in [0,1]$ interpolates generation/control and $\gamma_{\text{risk}} \ge 0$ is risk aversion.

**BAOAB Coefficients:**

$$
c_1 = e^{-\gamma h}, \qquad c_2 = \sqrt{(1 - c_1^2)\,T_c}

$$
*Cross-reference:* The boundary-reached condition is monitored by **[Node 25 (HoloGenCheck)](#node-25)** defined in {ref}`Section 21.4 <sec-summary-and-diagnostic-node>`.

(node-26)=
**Node 26: GeodesicCheck**

| **#**  | **Name**          | **Component**            | **Type**                   | **Interpretation**                    | **Proxy**                                                                                  | **Cost**  |
|--------|-------------------|--------------------------|----------------------------|---------------------------------------|--------------------------------------------------------------------------------------------|-----------|
| **26** | **GeodesicCheck** | **World Model / Policy** | **Trajectory Consistency** | Is trajectory approximately geodesic? | $\lVert\ddot{z} + \Gamma(\dot{z},\dot{z}) + G^{-1}\nabla\Phi_{\text{eff}} - u_\pi - \beta_{\text{curl}} G^{-1}\mathcal{F}\dot{z}\rVert_G$ | $O(BZ^2)$ |

**Trigger conditions:**
- High GeodesicCheck: Trajectory deviates from controlled geodesic (unexpected forces or integration errors).
- Remedy: Reduce time step $h$; verify Christoffel computation; check metric consistency.

(node-27)=
**Node 27: OverdampedCheck**

| **#**  | **Name**            | **Component** | **Type**            | **Interpretation**              | **Proxy**                                             | **Cost** |
|--------|---------------------|---------------|---------------------|---------------------------------|-------------------------------------------------------|----------|
| **27** | **OverdampedCheck** | **Policy**    | **Regime Validity** | Is friction dominating inertia? | $\gamma / \lVert \mathcal{M}_\gamma^{-1}\,v\rVert$ | $O(BZ)$  |

Here $v := \dot{z}$ and $\mathcal{M}_\gamma^{-1} = \gamma I - \beta_{\text{curl}} G^{-1}\mathcal{F}$.

**Trigger conditions:**
- Low OverdampedCheck: Operating in inertial regime; use full BAOAB integrator.
- Remedy: Increase friction $\gamma$ if overdamped limit desired; otherwise switch to second-order integrator.

(node-28)=
**Node 28: JumpConsistencyCheck**

| **#**  | **Name**                 | **Component**   | **Type**        | **Interpretation**                  | **Proxy**                                                       | **Cost**  |
|--------|--------------------------|-----------------|-----------------|-------------------------------------|-----------------------------------------------------------------|-----------|
| **28** | **JumpConsistencyCheck** | **World Model** | **WFR Balance** | Are jump rates consistent with WFR? | $\lvert\sum_j \lambda_{K\to j} - \sum_i \lambda_{i\to K}\rvert$ | $O(BK^2)$ |

**Trigger conditions:**
- High JumpConsistencyCheck: Jump rates violate detailed balance; may cause mass accumulation/depletion.
- Remedy: Recalibrate jump rates; verify value function consistency across charts.

(node-29)=
**Node 29: TextureFirewallCheck**

| **#**  | **Name**                 | **Component**  | **Type**                     | **Interpretation**              | **Proxy**                                       | **Cost**             |
|--------|--------------------------|----------------|------------------------------|---------------------------------|-------------------------------------------------|----------------------|
| **29** | **TextureFirewallCheck** | **Generation** | **Bulk-Boundary Separation** | Is texture decoupled from bulk? | $\lVert\partial_{z_{\text{tex}}} \dot{z}\rVert$ | $O(BZ_{\text{tex}})$ |

**Trigger conditions:**
- High TextureFirewallCheck: Texture is leaking into dynamics (firewall violated).
- Remedy: Review implementation; ensure texture sampled only at boundary; verify Axiom {prf:ref}`ax-bulk-boundary-decoupling`.

:::{div} feynman-prose
We have now assembled the complete dynamical picture. The agent is a particle on a curved manifold, carrying mass (belief weight), feeling forces (from the effective potential), being jostled by noise (exploration), and occasionally jumping between charts (conceptual transitions).

The beautiful thing is how much emerges from the geometry. The caution in risky regions? That is Mass = Metric. The exploration-exploitation tradeoff? That is the Einstein relation. The boundary behavior? That is the metric divergence. The phase transitions? That is the natural temperature schedule.

In the next sections, we will see how this dynamical picture connects to the boundary structure and the decoder. But the core message of this chapter is simple: **once you have the geometry right, the dynamics follow**.
:::
