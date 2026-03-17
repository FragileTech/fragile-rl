(sec-radial-generation-entropic-drift-and-policy-control)=
# Radial Generation: Entropic Drift and Policy Control

## TLDR

- Model “generation” as **radial flow**: start near a symmetric, high-entropy center and move outward toward committed,
  specific states.
- Two forces drive the flow: **entropic drift** (push outward) and **policy/value control** (choose direction).
- This unifies diffusion-style generation and RL control as the same geometric picture on the latent manifold.
- The practical outputs are monitorable diagnostics (RadialGenCheck / HoloGenCheck) and knob interpretations (horizon,
  cutoff radius, temperature).
- This chapter links WFR belief geometry to concrete generative/control behavior at the boundary.

## Roadmap

1. Motivation and the radial-flow picture.
2. The drift/control decomposition and its relation to diffusion models.
3. Diagnostics and practical failure modes (early stop vs. runaway drift).

:::{div} feynman-prose
Let me tell you what generation really is. Not the mathematical abstraction---that's coming---but the *idea* first.

Imagine you're standing at the center of a strange kind of universe. At the center, where you are, everything is possible. You could go anywhere, do anything, say anything. There's perfect symmetry in all directions. But the moment you take a step---any step---you've committed. You've broken that beautiful symmetry and started down a particular path.

As you walk outward from the center, two things happen. First, there's a force pushing you outward---call it "entropic drift." The universe wants you to move toward the boundary because there's just *more room* out there. (We'll see exactly why in a moment.) Second, if you have a goal---a task to accomplish, a sentence to complete, an action to take---that goal gives you a *direction*. It tells you which way to walk.

That's it. That's generation. You start at the center (maximum uncertainty, all possibilities open), and you flow outward toward the boundary (specific commitment, concrete output). The policy picks the direction; entropy provides the engine.

Now here's the beautiful part: this same picture unifies reinforcement learning and generative modeling. In RL, your "direction" comes from the value function---you walk toward high-reward regions. In generation, your "direction" comes from the prompt---you walk toward outputs that match the conditioning. Same geometry, same dynamics, different choice of which way to point.
:::

{cite}`ho2020ddpm,sohldickstein2015deep,nickel2017poincare`

(rb-diffusion-generation)=
:::{admonition} Researcher Bridge: Diffusion-Style Generation with Policy Drift
:class: info
If you know diffusion or score-based models, the radial expansion here is the generative flow. The policy is the controllable drift term that steers generation toward high-value regions.
:::

Data generation is defined as **radial expansion** of the latent state from the low-entropy origin ($z=0$) toward the high-entropy boundary ($|z| \to 1$). This boundary corresponds to the agent's {prf:ref}`def-boundary-markov-blanket`. The expansion is driven by the **entropic drift** (the natural tendency of {prf:ref}`def-hyperbolic-volume-growth` to increase) and steered by the **policy control field** $u_\pi$.

This section establishes the following unification: by identifying the **policy** as the source of initial direction selection, we merge Generative Modeling and Reinforcement Learning into a single variational operation:
- **RL:** The policy chooses a direction to maximize value $V(z)$.
- **Generation:** The policy (or context) chooses a direction to maximize semantic alignment with conditioning.
- Both contribute to the drift term in the latent SDE.

(sec-hyperbolic-volume-and-entropic-drift)=
## Hyperbolic Volume and Entropic Drift

:::{div} feynman-prose
Now I need to tell you something surprising about the geometry we're working in. You're probably used to Euclidean space---flat, ordinary space where a circle of radius $r$ has circumference $2\pi r$ and area $\pi r^2$. In Euclidean space, volume grows polynomially. Double the radius, quadruple the area.

But we're not in Euclidean space. We're in hyperbolic space, specifically the Poincare disk model. And in hyperbolic space, something much more dramatic happens: volume grows *exponentially* with radius.

Think about what this means. Near the center of the disk, there's relatively little "room"---few distinguishable states, few microstates, low entropy. But as you move toward the boundary, the amount of room explodes exponentially. There's vastly more space near the boundary than near the center.

This exponential growth is why there's an entropic force pushing you outward. Nature loves high-entropy configurations. If you randomly throw a particle into hyperbolic space, it will almost surely end up near the boundary, simply because there's so much more room there. The "entropic drift" we're about to define is just this statistical tendency made precise.
:::

Consider the latent agent as a particle in the Poincare disk $\mathbb{D} = \{z \in \mathbb{C} : |z| < 1\}$. The number of distinguishable microstates (volume) grows exponentially with radius $r$.

:::{prf:definition} Manifold Boundary and Interior
:label: def-manifold-boundary-and-interior

Let $\mathcal{Z}$ be the latent manifold with Poincare disk model. The **boundary** is the $(n-1)$-dimensional limit set:

$$
\partial\mathcal{Z} := \{z \in \mathbb{C}^n : |z| = 1\}.

$$
The **interior** (or bulk) is the open disk:

$$
\text{int}(\mathcal{Z}) := \{z \in \mathbb{C}^n : |z| < 1\}.

$$
These are standard differential geometry terms; the boundary is the ideal boundary at infinity in the hyperbolic metric.

:::

:::{div} feynman-prose
Notice something subtle here: the boundary is at $|z| = 1$, but in hyperbolic geometry, that boundary is infinitely far away! If you try to walk to it in the hyperbolic metric, you'll never get there---distances diverge as you approach. This is why it's called the "ideal boundary" or "boundary at infinity."

This isn't a bug; it's a feature. It means there's infinite capacity near the boundary to encode fine-grained distinctions, while the interior remains finite and manageable for planning.
:::

:::{prf:definition} Hyperbolic Volume Growth
:label: def-hyperbolic-volume-growth

With metric $G_{ij} = \frac{4\delta_{ij}}{(1-|z|^2)^2}$, the volume of a hyperbolic ball $B_r(0)$ grows exponentially:

$$
\mathrm{Vol}(B_r(0)) = 4\pi \sinh^2\!\left(\frac{r}{2}\right) \;\approx\; \pi e^r \quad \text{as } r \to \infty.

$$
Units: $[\mathrm{Vol}] = [z]^2$.

:::

:::{admonition} Why Exponential Growth Matters
:class: feynman-added note

Let me make the exponential growth concrete. In ordinary (Euclidean) 2D space:
- Circle of radius 1: area $\pi$
- Circle of radius 2: area $4\pi$
- Circle of radius 10: area $100\pi$

In hyperbolic space:
- Ball of radius 1: volume $\approx 4\pi \cdot 0.27 \approx 3.4$
- Ball of radius 2: volume $\approx 4\pi \cdot 1.38 \approx 17$
- Ball of radius 10: volume $\approx \pi e^{10} \approx 70,000$

See the difference? In hyperbolic space, going from radius 2 to radius 10 increases volume by a factor of about 4,000. In Euclidean space, it's only a factor of 25. This exponential growth is why hyperbolic space is so good for hierarchical representations---each level down the hierarchy can have exponentially more detail.
:::

:::{prf:definition} The Entropic Force
:label: def-the-entropic-force

The "Free Energy" of a state at radius $r$ is dominated by the entropic volume term $S(r) = 2 \operatorname{artanh}(r)$. To maximize entropy (fill the capacity), the agent experiences a radial force:

$$
F_{\text{entropy}}(z) = \nabla_G S(z) = \frac{(1-|z|^2)}{2} \cdot \frac{z}{|z|}

$$
This accounts for the Poincaré metric conformal factor. The drift magnitude decreases near the boundary ($|z| \to 1$), ensuring the agent asymptotically approaches but never reaches it.

Units: $[F_{\text{entropy}}] = [z]/\tau$.

:::

:::{div} feynman-prose
Here's the physical intuition. Imagine a gas molecule in a container shaped like the hyperbolic disk. If the molecule is near the center, there are relatively few places it could be. If it's near the boundary, there are exponentially more places. If you wait long enough and watch where the molecule spends its time, you'll find it near the boundary almost always---not because anything is pushing it there mechanically, but because that's where the room is.

The "entropic force" $F_{\text{entropy}}$ is just the gradient of this statistical tendency. It's not a real force in the Newtonian sense; it's an emergent force from the statistics of the underlying geometry. But it behaves like a force: it causes deterministic drift in the expected trajectory.
:::

:::{prf:proposition} Isotropic Radial Expansion
:label: prop-isotropic-radial-expansion

If acting alone (no policy steering), the entropic drift produces the isotropic expansion:

$$
r(\tau) = \tanh(\tau/2)

$$
This represents isotropic diffusion---expanding uniformly in all directions.

*Proof.* The overdamped equation $\dot{r} = (1-r^2)/2$ (from the Riemannian gradient of $U(z) = -2\operatorname{artanh}(|z|)$) integrates to $r(\tau) = \tanh(\tau/2 + \operatorname{artanh}(r_0))$. For $r_0 = 0$, we get $r(\tau) = \tanh(\tau/2)$. $\square$

:::

:::{admonition} What Does $r(\tau) = \tanh(\tau/2)$ Look Like?
:class: feynman-added example

Let's trace out this trajectory:

| Time $\tau$ | Radius $r = \tanh(\tau/2)$ | Interpretation |
|-------------|---------------------------|----------------|
| 0 | 0 | Starting at origin (vacuum) |
| 1 | 0.46 | Nearly halfway to boundary |
| 2 | 0.76 | Three-quarters out |
| 4 | 0.96 | Very close to boundary |
| $\infty$ | 1 | At boundary (never quite reached) |

Notice how the particle starts fast and slows down asymptotically. This makes sense: the $(1-r^2)/2$ term in the dynamics goes to zero as $r \to 1$. You never quite reach the boundary in finite time---which is appropriate, since the boundary is "at infinity" in hyperbolic terms.
:::

:::{prf:definition} Hyperbolic Information Potential
:label: def-hyperbolic-information-potential

The **information potential** $U: \mathbb{D} \to \mathbb{R}$ is the negative hyperbolic distance from the origin:

$$
U(z) := -d_{\mathbb{D}}(0, z) = -2 \operatorname{artanh}(|z|) = -\log\!\left(\frac{1+|z|}{1-|z|}\right).

$$
Units: $[U] = \mathrm{nat}$.

*Remark (Thermodynamic Interpretation).* At origin ($z=0$): $U = 0$ (maximum potential, maximum entropy). At boundary ($|z| \to 1$): $U \to -\infty$ (minimum potential, fully specified). The depth $-U(z)$ measures the **information content** of the state.

:::

:::{div} feynman-prose
Now this is elegant. The "information potential" $U(z)$ is just the negative distance from the origin. Think about what this means:

- At the origin, $U = 0$. Zero information content. Maximum entropy. All possibilities open.
- As you move outward, $U$ becomes more negative. You're "falling" down the information potential, committing to more specific states.
- At the boundary, $U \to -\infty$. You've committed everything. The state is fully specified.

This is exactly backwards from how we usually think about potential energy in physics (balls roll downhill). But it makes perfect sense for information: generating specific content requires "spending" your entropy budget. The potential $U$ tracks how much you've spent.
:::

:::{prf:proposition} Riemannian Gradient of $U$
:label: prop-riemannian-gradient-of

The gradient in the Poincaré metric is:

$$
\nabla_G U(z) = G^{-1} \nabla U = -\frac{(1-|z|^2)}{2} \hat{z}, \quad \text{where } \hat{z} = \frac{z}{|z|}.

$$
The **entropic drift** (negative gradient) pushes radially outward:

$$
-\nabla_G U(z) = \frac{(1-|z|^2)}{2} \hat{z}.

$$
*Remark (Connection to {ref}`Section 7.11 <sec-the-geometry-of-the-latent-space-a-hyperbolic-hierarchy>`).* The Poincare coordinate $z$ relates to depth via $\rho = d_{\mathbb{D}}(0, z) = 2\operatorname{artanh}(|z|)$. Chart transitions are handled by the WFR jump process ({ref}`Section 22.2 <sec-the-coupled-jump-diffusion-sde>`), governed by the {prf:ref}`def-the-wfr-action`.

**Cross-references:** Definition {prf:ref}`def-information-density-and-bulk-information-volume`, Theorem {prf:ref}`thm-capacity-constrained-metric-law`.

:::

:::{div} feynman-prose
Notice the $(1-|z|^2)/2$ factor in the entropic drift. Near the origin ($|z| \approx 0$), this is about $1/2$---a decent push outward. Near the boundary ($|z| \to 1$), this goes to zero. The drift weakens as you approach the boundary, which is why you asymptotically approach but never reach it.

This is the geometry telling you something important: the early stages of generation (when you're near the origin) happen quickly with strong drift. The final refinement (near the boundary) happens slowly and carefully. Coarse structure crystallizes fast; fine details take time.
:::

(sec-policy-control-field)=
## Policy Control Field

:::{div} feynman-prose
So far we've talked about the entropic drift---the force that pushes you outward from the origin. But that force is *radial*. It pushes you outward, but it doesn't tell you *which direction* to go outward. At the origin, every direction looks the same.

This is where the policy comes in.

At the origin, you have perfect rotational symmetry---$SO(D)$ symmetry in $D$ dimensions. Any direction is as good as any other. The policy breaks this symmetry. It picks a direction. And once that direction is picked, the entropic drift carries you along it toward the boundary.

Think of it like this: you're at the top of a perfectly symmetric mountain (the origin). Rolling down will take you to the bottom (the boundary), but you have to choose which face of the mountain to roll down. The policy is the initial kick that picks the face.
:::

At the origin ($z=0$), the system has full rotational symmetry $SO(D)$. To generate specific content (or solve a task), this symmetry must be broken. The **policy** provides the initial direction via the control field $u_\pi$.

:::{prf:proposition} SO(D) Symmetry at Origin
:label: prop-so-d-symmetry-at-origin

At $z = 0$:
1. The metric is isotropic: $G(0) = 4I$
2. The entropic force vanishes: $F_{\text{entropy}}(0) = 0$
3. The system has full rotational symmetry $SO(D)$

*Cross-reference (Gauge Breaking):* This $SO(D)$ symmetry is the special case where the stabilizer subgroup $H_0 = \{e\}$ is trivial. In multi-agent settings, this symmetry is spontaneously broken via the Higgs mechanism (Theorem {prf:ref}`thm-higgs-mechanism`), yielding massive gauge bosons and effective agent masses.

:::

:::{admonition} Why Does the Entropic Force Vanish at the Origin?
:class: feynman-added note

Look back at the formula: $F_{\text{entropy}}(z) = \frac{1-|z|^2}{2}\hat{z}$. At $z = 0$, this gives exactly zero (the unit vector $\hat{z}$ is undefined, but the prefactor vanishes anyway).

Physically, this makes sense. The entropic force points radially outward, but at the exact center, there is no "radial direction"---all directions are equivalent. The force has to vanish there by symmetry.

This is crucial for the policy's role: at the origin, the policy is the *only* thing that provides a direction. The system waits for the policy's kick before it knows which way to go.
:::

:::{prf:definition} The Control Field
:label: def-the-control-field

The Policy $\pi_\theta(a|z)$ outputs a **control field** $u_\pi(z)$ on the tangent bundle $T\mathbb{D}$:

$$
u_\pi(z) = G^{-1}(z) \cdot \mathbb{E}_{a \sim \pi_\theta}[a]

$$
This vector field represents the **Information Preference** of the agent (or the User).

Units: $[u_\pi] = [z]/\tau$.

*Remark (Context-Conditioning).* {ref}`Section 23.6 <sec-relationship-to-the-context-conditioned-framework>` generalizes this to **context-conditioned policies** $\pi(a|z,c)$ where the context $c \in \mathcal{C}$ unifies: RL action spaces, classification label spaces, and LLM prompt spaces. The control field becomes $u_\pi(z,c) = G^{-1}(z) \cdot \nabla_z \Phi_{\text{eff}}(z,K,c)$ where the {prf:ref}`def-effective-potential` depends on task context.

:::

:::{div} feynman-prose
The $G^{-1}(z)$ factor is important---it converts the policy's "raw action" into a proper tangent vector in the curved geometry. Without this metric correction, the policy's influence would be distorted by the geometry.

Near the origin, $G^{-1}(0) = I/4$, so the control field is just a scaled version of the policy's expected action. Near the boundary, $G^{-1}$ goes to zero, which means the policy's influence weakens as you get more committed. This is appropriate: early in generation, you want strong steering; late in generation, the trajectory is mostly determined.
:::

:::{prf:definition} Control Field at Origin
:label: def-control-field-at-origin

At $\tau=0$, the total drift is:

$$
F_{\text{total}} = F_{\text{entropy}} + u_\pi(0)

$$
Since $F_{\text{entropy}}(0) = 0$ (isotropic), the initial trajectory is determined **entirely** by $u_\pi(0)$.

:::

:::{div} feynman-prose
This is the key insight: **the policy's only essential job is to pick the initial direction**. Once you've moved away from the origin, the entropic drift takes over and carries you toward the boundary. The policy can keep adjusting the direction along the way, but the big decision---which semantic region to head toward---is made at the very beginning.

This explains why prompts matter so much in language models, and why initial state initialization matters so much in RL. The first moment of generation is when the crucial symmetry-breaking happens.
:::

:::{prf:theorem} Unified Control Interpretation
:label: thm-unified-control-interpretation

The control field $u_\pi$ admits three equivalent interpretations:

| **Mode**                     | **Control Field $u_\pi$**                          | **Interpretation**                         |
|------------------------------|----------------------------------------------------|--------------------------------------------|
| **RL**                       | $u_\pi = G^{-1} \nabla_z V_{\text{critic}}$        | Points toward high-value regions           |
| **Conditioned Generation**   | $u_\pi = G^{-1} \cdot \text{embed}(\text{prompt})$ | Clamped to user's prompt embedding         |
| **Unconditional (Dreaming)** | $u_\pi = 0$                                        | Pure thermal fluctuation selects direction |

*Proof.* In all cases, $u_\pi$ is a tangent vector at $z$. The RL case follows from the policy gradient theorem {cite}`sutton1999policy`; the generation case follows from treating the prompt as a target direction; the unconditional case reduces to pure Langevin dynamics where noise breaks symmetry. $\square$

:::

:::{admonition} The Three Modes in Plain English
:class: feynman-added example

**RL mode:** "I want to go where the reward is." The value function $V$ tells you which regions of latent space are valuable. The policy points uphill on this value landscape.

**Generation mode:** "I want to match this prompt." The prompt gets embedded as a direction in latent space. The policy points toward that direction, regardless of reward.

**Dreaming mode:** "I have no goal, just let me wander." The policy contributes nothing; only thermal noise picks a direction. This is like unguided imagination or free association.

The beautiful thing is that all three are just different settings of the same dial: the control field $u_\pi$. The geometry doesn't care why you picked a direction---it just carries you along once you've picked.
:::

:::{prf:theorem} Angular Symmetry Breaking {cite}`strogatz2015nonlinear`
:label: thm-angular-symmetry-breaking

In the **overdamped limit** of the second-order geodesic Langevin equation (Definition {prf:ref}`def-bulk-drift-continuous-flow`, Theorem {prf:ref}`thm-overdamped-limit`), the generation dynamics decompose into radial expansion and angular symmetry breaking.

**Radial dynamics (monotonic expansion):** The radial coordinate $r = |z|$ satisfies:

$$
dr = \frac{1-r^2}{2}\,d\tau + \frac{1-r^2}{2}\sqrt{2T_c}\,dW_r

$$
with drift $\frac{1-r^2}{2} > 0$ for all $r \in [0,1)$. The origin is not a fixed point; the drift pushes trajectories outward (though stochastic fluctuations can temporarily reverse this at small $r$).

**Angular dynamics (symmetry breaking):** In polar coordinates $z = re^{i\theta}$, the angular evolution satisfies:

$$
d\theta = \frac{u_\pi^\theta}{r}\,d\tau + \frac{1-r^2}{2r}\sqrt{2T_c}\,dW_\theta

$$
where $u_\pi^\theta = u_\pi \cdot \hat{\theta}$ is the tangential component of the control field.

**Phase Transition:** The $SO(D)$ rotational symmetry undergoes spontaneous breaking controlled by the dimensionless ratio:

$$
\eta(r) := \frac{|u_\pi^\theta|^2}{T_c} \cdot \frac{2r^2}{(1-r^2)^2}

$$
- **Symmetric phase** ($\eta \ll 1$): Angular noise dominates; direction randomizes
- **Broken phase** ($\eta \gg 1$): Policy dominates; direction determined by $u_\pi$

*Proof.* Starting from the second-order geodesic Langevin equation (Definition {prf:ref}`def-bulk-drift-continuous-flow`) with the Poincaré metric $G_{ij} = \frac{4\delta_{ij}}{(1-r^2)^2}$, we take the overdamped limit (Theorem {prf:ref}`thm-overdamped-limit`). The overdamped position SDE in Cartesian coordinates is:

$$
dz^k = -G^{kj}\partial_j U\, d\tau + u_\pi^k\, d\tau + \sqrt{2T_c}(G^{-1/2})^{kj}\,dW^j_\tau

$$
where $G^{-1/2} = \frac{1-r^2}{2}I$. Converting to polar coordinates via Itô's lemma:
- Radial: $dr = \langle dz, \hat{r}\rangle + \frac{1}{2}\text{tr}(\text{Hess}_r \cdot \Sigma)$ where $\Sigma = 2T_c G^{-1}$
- Angular: $d\theta = \langle dz, \hat{\theta}/r\rangle + \frac{1}{2}\text{tr}(\text{Hess}_\theta \cdot \Sigma)$

The Itô corrections vanish for the radial component (since $\partial^2 r/\partial z^i\partial z^j$ is traceless) and contribute a drift correction for angular that cancels with geometric terms. The stated SDEs follow after simplification.

**Critical temperature:** The symmetry-breaking ratio $\eta(r)$ compares the squared angular drift to the angular diffusion coefficient. At characteristic radius $r_*$, setting $\eta(r_*) = 1$ defines the critical temperature.

**Direction freeze-out:** As $r$ increases toward the boundary, $\eta(r) \to \infty$ (the denominator $(1-r^2)^2 \to 0$), causing the angular distribution to concentrate. The direction selected at early times persists to the boundary. $\square$

:::

:::{div} feynman-prose
Let me explain why the radial and angular dynamics separate so cleanly.

The radial direction is driven by entropy: the system always wants to expand outward because there is exponentially more "room" near the boundary than near the origin (recall the hyperbolic volume growth from Definition {prf:ref}`def-hyperbolic-volume-growth`). This expansion is inexorable---there is no fixed point at the origin, no moment of hesitation. The trajectory begins moving outward immediately.

The angular direction is where the interesting physics happens. At the origin, all directions are equivalent by $SO(D)$ symmetry. As the trajectory expands, it must "choose" a direction. This choice is governed by the competition between two forces:
1. The **policy** $u_\pi^\theta$, which nudges the trajectory toward a preferred direction
2. **Thermal noise**, which randomizes the direction

Near the origin (small $r$), the angular noise dominates because the noise coefficient scales as $1/r$. As the trajectory expands (larger $r$), the noise weakens relative to the policy, and the direction "freezes in."

The critical temperature $T_c^*$ marks the boundary between two regimes: below it, the policy wins and generation is coherent; above it, noise wins and generation is incoherent.
:::

:::{admonition} Temperature and Generation Quality
:class: feynman-added warning

This explains the familiar phenomenon of "temperature" in language models:
- **Low temperature** ($T_c < T_c^*$): Sharp, coherent outputs. The policy determines the direction early; the trajectory follows a nearly deterministic path to the boundary.
- **High temperature** ($T_c > T_c^*$): Diffuse, incoherent outputs. Angular noise dominates; the direction wanders randomly throughout generation.

The angular symmetry breaking (Theorem {prf:ref}`thm-angular-symmetry-breaking`) makes this precise: there is a critical temperature below which the policy controls direction selection, and above which thermal fluctuations dominate.
:::

(pi-symmetry-breaking)=
::::{admonition} Physics Isomorphism: Spontaneous Symmetry Breaking
:class: note

**In Physics:** Spontaneous symmetry breaking occurs when a system's ground state has lower symmetry than its Hamiltonian. The classic example is the Mexican hat potential $V(\phi) = -\mu^2|\phi|^2 + \lambda|\phi|^4$: for $\mu^2 > 0$, the $U(1)$-symmetric origin becomes unstable and the system selects a direction {cite}`goldstone1961field,weinberg1996qft`.

**In Implementation:** The angular dynamics exhibit symmetry breaking (Theorem {prf:ref}`thm-angular-symmetry-breaking`):

$$
d\theta = \frac{u_\pi^\theta}{r}\,d\tau + \frac{\sqrt{T_c(1-r^2)}}{r}\,dW_\theta

$$
The $SO(D)$ symmetry is broken when the policy-to-noise ratio $\eta(r) = |u_\pi^\theta|^2 r^2 / [T_c(1-r^2)]$ exceeds unity.

**Correspondence Table:**
| Phase Transition Theory | Agent (Policy Emergence) |
|:------------------------|:-------------------------|
| Order parameter $\phi$ | Angular direction $\theta$ |
| Control parameter | Policy strength $|u_\pi^\theta|$ |
| Critical temperature $T_c^*$ | $T_c^* \approx |u_\pi^\theta|^2 r_*^2$ |
| Symmetric phase | Isotropic angular distribution |
| Broken phase | Policy-selected direction $\theta_\pi$ |
| Goldstone modes | Angular fluctuations in $\theta$ |

**Significance:** Policy selection is not arbitrary---it is a geometric phase transition where the agent spontaneously breaks $SO(D)$ symmetry to select a generation direction.
::::

:::{div} feynman-prose
The Goldstone modes are worth a moment's thought. When you break a continuous symmetry (like $SO(D)$), you don't get to keep the original degrees of freedom for free. Some of them turn into "Goldstone modes"---massless excitations that correspond to directions along which the symmetry *could have been broken differently*.

In our case, once you've picked a direction to generate (broken the $SO(D)$ symmetry), the Goldstone modes are the angular fluctuations around that direction. These are the "variations on a theme"---different ways of generating content that are semantically nearby but not identical.

This is why you can sample from a language model multiple times with the same prompt and get different but related outputs. You're exploring the Goldstone manifold around a particular symmetry-breaking direction.
:::

**Algorithm 21.2.6 (Control Field Computation).**

```python
import torch
from typing import Literal, Optional


def poincare_metric_inv(z: torch.Tensor) -> torch.Tensor:
    """Compute inverse Poincare metric G^{-1}(z) = (1 - |z|^2)^2 / 4."""
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    one_minus_r_sq = torch.clamp(1.0 - r_sq, min=1e-8)
    return (one_minus_r_sq ** 2) / 4.0


def compute_control_field(
    z: torch.Tensor,                    # [B, D] current position (near origin)
    mode: Literal["rl", "generation", "dreaming"],
    prompt_embed: Optional[torch.Tensor] = None,  # [B, D] for generation mode
    grad_V: Optional[torch.Tensor] = None,        # [B, D] critic gradient for RL
    T_c: float = 1.0,                   # Temperature (for dreaming mode)
) -> torch.Tensor:
    """
    Compute the control field u_pi that selects initial direction.

    Breaks SO(D) symmetry at the origin, unifying RL, generation,
    and dreaming into a single operation.

    Cross-ref: Theorem {prf:ref}`thm-unified-control-interpretation`
    """
    B, D = z.shape
    G_inv = poincare_metric_inv(z)  # [B, 1]

    if mode == "rl":
        # RL: u_pi = G^{-1} * grad V (points toward high value)
        if grad_V is None:
            raise ValueError("grad_V required for RL mode")
        u_pi = G_inv * grad_V

    elif mode == "generation":
        # Generation: u_pi = G^{-1} * prompt_embed
        if prompt_embed is None:
            raise ValueError("prompt_embed required for generation mode")
        # Normalize prompt to unit vector, then scale by metric
        prompt_norm = prompt_embed / (torch.norm(prompt_embed, dim=-1, keepdim=True) + 1e-8)
        u_pi = G_inv * prompt_norm

    elif mode == "dreaming":
        # Dreaming: pure thermal fluctuation (no deterministic kick)
        # The noise in the Langevin dynamics will break symmetry
        u_pi = torch.zeros(B, D, device=z.device, dtype=z.dtype)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return u_pi
```

**Cross-references:** {ref}`Section 2.7 <sec-the-hjb-correspondence>` (HJB Correspondence), Section 14.2 (MaxEnt control equivalence).

:::

::::{admonition} Connection to RL #24: Diffusion Policies as Degenerate Radial Generation
:class: note
:name: conn-rl-24
**The General Law (Fragile Agent):**
Data generation is **radial expansion** from the vacuum (origin) to the boundary:

$$
F_{\text{total}} = \underbrace{F_{\text{entropy}}}_{\text{Hyperbolic drift}} + \underbrace{u_\pi}_{\text{Policy kick}}

$$
where $F_{\text{entropy}} = \nabla_G S(z)$ is entropic drift from hyperbolic volume growth and $u_\pi = G^{-1} \nabla_z V$ is the policy control field that breaks $SO(D)$ symmetry.

**The Degenerate Limit:**
Replace hyperbolic geometry with Euclidean ($G \to I$). Reverse the direction (boundary to origin). Remove value-based steering.

**The Special Case (Standard RL):**

$$
dz_t = s_\theta(z_t, t)\, dt + \sigma(t)\, dW_t, \quad z_T \sim \mathcal{N}(0, I)

$$
This recovers **Diffusion Models** {cite}`ho2020ddpm` and **Diffusion Policies** for robotic control.

**What the generalization offers:**
- **Hyperbolic structure**: Exponential volume growth provides natural hierarchy (Theorem **Thm: Hyperbolic Volume Growth**)
- **Forward generation**: Origin to boundary matches RL's forward dynamics semantics
- **Policy unification**: RL control and conditional generation share the same drift term $u_\pi$
- **Symmetry breaking**: Policy kicks at origin select generation mode (Theorem {prf:ref}`thm-angular-symmetry-breaking`)
::::

(sec-bulk-boundary-independence)=
## Bulk-Boundary Independence

:::{div} feynman-prose
Now I want to tell you about a separation that's absolutely crucial to how this whole thing works: the separation between the **bulk** (the interior of the disk, where planning happens) and the **boundary** (the edge of the disk, where observation and action happen).

Here's the key idea: when you're planning a trajectory---thinking about what to do, imagining futures, computing values---you're operating in the bulk. You're moving around the interior of the latent space, figuring out which direction to go. During this planning phase, you don't need to worry about the fine details of what things look like or sound like. You're thinking at the level of structure, causality, and value.

It's only when you reach the boundary---when you commit to an output and need to actually generate pixels, tokens, or motor commands---that the fine-grained details matter. That's where "texture" comes in: the high-frequency variations that make things look real but don't affect the underlying meaning or decision.

This separation has a beautiful consequence: the planning process doesn't waste capacity on texture. It focuses purely on what matters for decision-making. Texture gets added at the very end, at the boundary, as a kind of cosmetic finishing step.
:::

We strictly enforce the separation of **Planning** (interior $\text{int}(\mathcal{Z})$) and **Observation** (boundary $\partial\mathcal{Z}$). This is formalized as a partition condition.

*Remark (Motor Extension).* The independence constraint applies equally to the **motor/action boundary**: motor texture $z_{\text{tex,motor}}$ (tremor, fine motor noise) is sampled at the output interface and does not participate in planning. {ref}`Section 23.3 <sec-motor-texture-the-action-residual>` formalizes the motor texture distribution $z_{\text{tex,motor}} \sim \mathcal{N}(0, \sigma_{\text{motor}}^2 G^{-1}(z))$ with the same conformal scaling as visual texture. Theorem {prf:ref}`ax-motor-texture-firewall` establishes the duality: $\Sigma_{\text{motor}} = \omega \cdot \Sigma_{\text{visual}} \cdot \omega^{-1}$.

:::{prf:axiom} Bulk-Boundary Decoupling
:label: ax-bulk-boundary-decoupling

The state decomposition $Z = (K, z_n, z_{\text{tex}})$ satisfies a **partition condition**:

1. **Interior (Planning Domain):** The trajectory $z(\tau)$ evolves strictly on the manifold $\mathcal{Z} = \mathcal{K} \times \mathcal{Z}_n$. It contains no texture component. Planning depends only on geometry and topology:

$$
\dot{z} = f(z, u_\pi) \quad (\text{No } z_{\text{tex}} \text{ dependence})

$$
2. **Boundary Interface:** Texture $z_{\text{tex}}$ is a stochastic component that exists **only** at the interface where the internal state meets the external observation:

$$
z_{\text{tex}} \sim \mathcal{N}(0, \Sigma(z_{\text{final}}))

$$
Formally, the partition condition is:

$$
\frac{\partial}{\partial z_{\text{tex}}} \left[ \dot{z}^k, \lambda_{\text{jump}}, u_\pi \right] = 0 \quad \forall \tau \in [0, \tau_{\text{stop}})

$$
:::

:::{admonition} The Partition in Plain Language
:class: feynman-added note

Think of it this way: there are two completely separate parts of the system.

**The Bulk (Planning):**
- Works with coarse-grained structure: "Is this a cat? Where is it going? Should I chase it?"
- Operates on $(K, z_n)$---discrete categories and continuous nuisance variables
- No texture involved. Clean, abstract reasoning.

**The Boundary (Interface):**
- Deals with fine-grained details: "What exact shade of orange is the fur? What's the precise pixel pattern?"
- Adds $z_{\text{tex}}$ only when generating actual outputs
- Stochastic: samples from a distribution, doesn't compute deterministically

The partition condition says: these two parts *never mix*. Planning doesn't see texture. Texture doesn't affect planning. They're firewalled from each other.

Why does this matter? Because it means you can do complex planning with limited capacity. You don't need to simulate every pixel to figure out whether to turn left or right. You only need to simulate the structure.
:::

:::{prf:definition} Boundary Texture Distribution
:label: def-boundary-texture-distribution

At the terminal position $z_{\text{final}}$, texture is sampled from a **geometry-dependent** Gaussian:

$$
z_{\text{tex}} \sim \mathcal{N}\big(0,\, \Sigma(z_{\text{final}})\big),

$$
where the covariance matrix is:

$$
\Sigma(z) = \sigma_{\text{tex}}^2 \cdot G^{-1}(z) = \sigma_{\text{tex}}^2 \cdot \frac{(1-|z|^2)^2}{4} I.

$$
Units: $[\Sigma] = [z_{\text{tex}}]^2$.

:::

:::{div} feynman-prose
This is a lovely formula. The texture variance $\Sigma(z)$ scales with $G^{-1}(z)$, the inverse metric. What does that mean?

Near the origin ($|z| \approx 0$), $G^{-1} \approx 1/4$, so texture has moderate variance. The output is coarse-grained, uncertain, blurry.

Near the boundary ($|z| \to 1$), $G^{-1} \to 0$, so texture variance goes to zero. The output is fine-grained, precise, sharp.

This is exactly the right behavior! When you're generating something abstract and uncertain, the texture should be noisy and uncommitted. When you're generating something specific and definite, the texture should be precise and deterministic.

The geometry *enforces* the correct texture statistics. You don't have to tune this by hand---it falls out automatically from the hyperbolic structure.
:::

:::{prf:proposition} Conformal Texture Scaling
:label: prop-conformal-texture-scaling

The texture variance scales with the inverse metric:

| **Region** | **$\lvert z\rvert$** | **$\Sigma(z)$**                            | **Interpretation**        |
|------------|----------------------|--------------------------------------------|---------------------------|
| Origin     | $\approx 0$          | $\sigma_{\text{tex}}^2/4 \cdot I$          | Moderate texture (coarse) |
| Mid-disk   | $\approx 0.5$        | $\sigma_{\text{tex}}^2 \cdot 9/64 \cdot I$ | Reduced texture           |
| Boundary   | $\to 1$              | $\to 0$                                    | Deterministic texture     |

*Remark (Conformal suppression).* Near the boundary (high resolution/specificity), the metric $G$ diverges, so $G^{-1} \to 0$ and texture fluctuations are suppressed.

:::

:::{admonition} Why Conformal Scaling is the Right Choice
:class: feynman-added tip

You might wonder: why specifically $\Sigma \propto G^{-1}$? Why not some other scaling?

The answer is that $G^{-1}$ is the *natural* choice for covariances in curved space. If you want your texture distribution to look "isotropic" to an observer using the hyperbolic metric, you need to scale the Euclidean covariance by $G^{-1}$. Otherwise, texture would look stretched or compressed depending on position.

Technically: a Gaussian $\mathcal{N}(0, G^{-1})$ is the maximum-entropy distribution subject to a constraint on expected hyperbolic distance from the mean. It's the "least assuming" distribution you can pick at each point.

So conformal scaling isn't just convenient---it's the unique geometrically natural choice.
:::

:::{prf:definition} Boundary Decoder
:label: def-boundary-decoder

The Decoder $\mathcal{D}$ is the **only** component that sees texture. It performs the **boundary synthesis**:

$$
x = \mathcal{D}(z_{\text{final}}, z_{\text{tex}})

$$
where:
- $z_{\text{final}} = (e_K, z_n)$: Determines the shape, physics, and causal structure
- $z_{\text{tex}}$: "Paints" the high-frequency details onto that structure

:::

:::{div} feynman-prose
I like to think of the decoder as a two-stage artist:

**Stage 1:** Draw the structure. Given $z_{\text{final}}$, the decoder knows *what* to draw---the bones, the shapes, the logical relationships. This is the "skeleton" of the output.

**Stage 2:** Paint the texture. Given $z_{\text{tex}}$, the decoder fills in the details---the colors, the patterns, the fine variations. This is the "skin" of the output.

Neither stage can function without the other. Structure without texture gives you a wireframe. Texture without structure gives you noise. The decoder combines them into a coherent output.

Notice that the planning system (which lives in the bulk) only affects Stage 1. It controls *what* gets generated but not the fine details of *how* it looks. This is the bulk-boundary decoupling in action.
:::

:::{prf:proposition} Epistemic Barrier
:label: prop-epistemic-barrier

The partition condition enforces **BarrierEpi** (Epistemic Limit): The agent does not waste capacity predicting the noise---it only predicts the *statistics* of the noise ($\Sigma$).

:::

:::{admonition} The Wisdom of Not Trying Too Hard
:class: feynman-added note

This proposition is philosophically important. It says the agent *shouldn't* try to predict texture precisely. Texture is, by design, unpredictable fine-grained variation. Trying to predict it would be wasteful.

Instead, the agent learns the *statistics* of texture: where it's noisy, where it's smooth, what the variance is at different positions. This is much more efficient than trying to predict exact values.

You see this pattern in biological systems too. Your visual system doesn't try to predict individual photon arrivals; it learns the statistics of images. Your motor system doesn't try to plan individual muscle fiber twitches; it plans movements and lets noise handle the details.

The partition condition formalizes this: there's a boundary between what's worth predicting (structure) and what's better left to statistics (texture).
:::

:::{prf:definition} Stopping Criterion
:label: def-stopping-criterion

The flow terminates when the radial coordinate exceeds a cutoff:

$$
\tau_{\text{stop}} := \inf\{\tau \ge 0 : |z(\tau)| \ge R_{\text{cutoff}}\}

$$
This is equivalent to the information stopping criterion $I_{\text{bulk}}(z) \ge C_\partial$ (Theorem {prf:ref}`thm-capacity-constrained-metric-law`).
In practice, choose $R_{\text{cutoff}} = 1 - \varepsilon$ with $\varepsilon$ tied to Levin length/resolution. This is a
computational cutoff, not a terminal task boundary.

**Algorithm 21.3.7 (Boundary Texture Sampling).**

```python
import torch

def sample_boundary_texture(
    z_final: torch.Tensor,        # [B, D] final semantic position
    texture_dim: int,             # Dimension of texture space
    sigma_tex: float = 1.0,       # Base texture std dev
) -> torch.Tensor:
    """
    Sample texture with geometry-dependent variance.

    Implements Definition 21.3.2:
        z_tex ~ N(0, Sigma(z_final))
        Sigma(z) = sigma_tex^2 * G^{-1}(z)

    The partition condition (Axiom 21.3.1) ensures this is called
    ONLY at terminal time, not during interior dynamics.

    Cross-ref: Proposition 21.3.3 (Conformal Scaling)
    """
    B = z_final.shape[0]

    # Compute G^{-1}(z) = (1 - |z|^2)^2 / 4
    r_sq = (z_final ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
    one_minus_r_sq = torch.clamp(1.0 - r_sq, min=1e-6)
    G_inv_scale = (one_minus_r_sq ** 2) / 4.0  # [B, 1]

    # Texture std = sigma_tex * sqrt(G^{-1})
    texture_std = sigma_tex * torch.sqrt(G_inv_scale)  # [B, 1]

    # Sample isotropic Gaussian, then scale
    z_tex = torch.randn(B, texture_dim, device=z_final.device)
    z_tex = z_tex * texture_std  # broadcast scaling

    return z_tex
```

:::

:::{div} feynman-prose
The stopping criterion $|z| \ge R_{\text{cutoff}}$ is your "resolution dial." Set $R_{\text{cutoff}}$ close to 1, and you generate very specific, detailed outputs. Set it smaller, and you generate coarser, more abstract outputs.

There's a tradeoff here: higher $R_{\text{cutoff}}$ means more computation (the trajectory takes longer to reach the boundary) but finer outputs. Lower $R_{\text{cutoff}}$ is faster but coarser. You tune this based on your application's needs.

For most generation tasks, you want $R_{\text{cutoff}}$ fairly high---maybe 0.95 or 0.99. For quick sketching or brainstorming, a lower value might suffice.
:::

(sec-summary-and-diagnostic-node)=
## Summary and Diagnostic Node

:::{div} feynman-prose
Let me step back and summarize what we've built in this section.

We have a picture of generation as **radial expansion in hyperbolic space**. Starting from the origin (maximum entropy, all possibilities), the agent flows outward toward the boundary (minimum entropy, specific commitment). This flow is driven by:

1. **Entropic drift:** The natural tendency of hyperbolic geometry to push mass outward, toward the exponentially larger boundary region.

2. **Policy control:** The direction of expansion, chosen by the policy to maximize value (in RL) or match conditioning (in generation).

At the origin, the agent faces perfect symmetry---any direction is equally valid. The policy breaks this symmetry via a phase transition (pitchfork bifurcation). Below a critical temperature, the agent commits to a direction and flows deterministically. Above it, the agent dithers randomly.

The trajectory itself happens in the **bulk**---the interior of the disk---where planning is clean and abstract. Only at the **boundary** does fine-grained texture get added, sampled from a geometry-dependent Gaussian that naturally suppresses noise near the boundary.

This is generation: break symmetry at the origin, ride the entropic flow toward the boundary, add texture at the end. RL and generative modeling are the same thing, just with different policies.
:::

**Summary of Radial Generation:**

| **Aspect**          | **Formula**                                   | **Units**            | **Reference**                                    |
|---------------------|-----------------------------------------------|----------------------|--------------------------------------------------|
| Entropic Drift      | $F_{\text{entropy}} = \frac{1-\lvert z\rvert^2}{2}\hat{z}$ | $[z]/\tau$           | Def {prf:ref}`def-the-entropic-force`            |
| Radial Expansion    | $r(\tau) = \tanh(\tau/2)$                     | dimensionless        | Prop {prf:ref}`prop-isotropic-radial-expansion`  |
| Control Field       | $u_\pi = G^{-1} \mathbb{E}[a]$                | $[z]/\tau$           | Def {prf:ref}`def-the-control-field`             |
| Partition Condition | $\partial_{z_{\text{tex}}} \dot{z} = 0$       | -                    | Axiom {prf:ref}`ax-bulk-boundary-decoupling`     |
| Texture Covariance  | $\Sigma(z) = \sigma_{\text{tex}}^2 G^{-1}(z)$ | $[z_{\text{tex}}]^2$ | Def {prf:ref}`def-boundary-texture-distribution` |
| Stopping            | $\lvert z\rvert \ge R_{\text{cutoff}}$        | dimensionless        | Def {prf:ref}`def-stopping-criterion`            |

(node-25)=
**Node 25: RadialGenCheck (HoloGenCheck)**

| **#**  | **Name**           | **Component** | **Type**                | **Interpretation**       | **Proxy**                                                         | **Cost** |
|--------|--------------------|---------------|-------------------------|--------------------------|-------------------------------------------------------------------|----------|
| **25** | **RadialGenCheck** | **Generator** | **Generation Validity** | Did flow reach boundary? | $\mathbb{I}(\lvert z_{\text{final}}\rvert \ge R_{\text{cutoff}})$ | $O(B)$   |

**Trigger conditions:**
- Low RadialGenCheck: Generation terminated too early (insufficient specificity).
- Remedy: Increase $\tau_{\text{max}}$ or decrease $R_{\text{cutoff}}$.

**Cross-references:** {ref}`Section 2.2b <sec-the-shutter-as-a-vq-vae>` (VQ-VAE texture channel), {ref}`Section 7.10 <sec-decoder-architecture-overview-topological-decoder>` (TopologicalDecoder), {ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>` (Capacity constraints).
