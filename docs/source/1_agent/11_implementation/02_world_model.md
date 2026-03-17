(sec-covariant-world-model)=
# The Covariant World Model: Geodesic Dynamics on the Poincare Ball

## TLDR

- The **covariant geometric world model** predicts future latent states by numerically integrating
  Hamiltonian dynamics on the Poincare ball, using a geodesic Boris-BAOAB splitting scheme that
  respects the hyperbolic geometry at every step.
- The WFR decomposition ({ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`)
  splits state evolution into two components: **continuous BAOAB flow** ($W_2$ transport within a chart)
  and **discrete chart jumps** (Fisher-Rao reaction between charts).
- All force computations --- potential, control, curl --- use **CovariantAttention**
  ({ref}`sec-covariant-cross-attention-architecture`) over chart and action tokens, ensuring
  gauge covariance with respect to the atlas structure learned by the encoder
  ({ref}`sec-shared-dynamics-encoder`).
- CFL stability bounds derived from a single **minimum length scale** $\ell_{\min}$ control
  force magnitude, velocity, and the conformal factor, preventing numerical blowup near the
  Poincare boundary.
- The **3-phase training pipeline** first trains the encoder alone (Phase 1), then trains the
  world model with a frozen encoder (Phase 2), and finally fine-tunes both jointly with
  alternating optimization and gradient surgery (Phase 3).
- Loss functions enforce six physics-inspired constraints: geodesic accuracy, chart prediction,
  momentum regularization, energy conservation, Hodge consistency, and the screened Poisson PDE.

## Roadmap

1. Why physics? The motivation for Hamiltonian mechanics on curved space.
2. Architecture overview: the modular sub-module design.
3. The covariant force field: analytic drive, learned critic, and risk forces.
4. The control field: how actions become geometry-respecting forces.
5. The geodesic Boris-BAOAB integrator: the complete integration step.
6. Conditional chart jumps: the Fisher-Rao component of WFR dynamics.
7. Loss functions: teaching the world model physics.
8. The three-phase training pipeline.
9. Diagnostics: reading the instruments.
10. Putting it all together: theory-to-code correspondence.


(sec-why-physics-for-world-model)=
## 1. Why Physics? The World Model as a Hamiltonian System

:::{div} feynman-prose
Now, you might think the simplest thing to do is take the encoder's latent representation and
feed it through a big MLP to predict the next state. And you would be right that it is the
simplest thing. But it is not the right thing.

Here is why. An MLP knows nothing about the space it operates in. It treats the latent as
a flat Euclidean vector and applies arbitrary nonlinear transformations. But our latent space
is the Poincare ball --- a curved space where distances grow exponentially near the boundary,
where parallel transport is nontrivial, where straight lines are arcs. If you ignore all this
structure and just throw an MLP at the problem, you are asking the network to *rediscover*
hyperbolic geometry from data. It can approximate it, sure, but it wastes capacity on something
we already know.

The alternative is to build the geometry in from the start. Instead of learning an arbitrary
function $z_{t+1} = f(z_t, a_t)$, we write down a Hamiltonian system on the Poincare ball and
learn only the *forces*. The integration scheme handles the geometry. The network handles the
physics. Each does what it is good at.

But there is an even deeper reason. The Wasserstein-Fisher-Rao geometry
({ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`) tells us
that state evolution has two fundamentally different components: smooth transport *within* a
chart, and discrete jumps *between* charts. An MLP mushes these together. Our world model
separates them: BAOAB handles the smooth flow, and a conditional jump process handles the
chart transitions. The theory says they should be separate, so the code keeps them separate.
:::

The `GeometricWorldModel` implements the unified WFR world model ({prf:ref}`def-wfr-world-model`)
as a concrete PyTorch module. The theoretical equations of motion
({ref}`sec-the-equations-of-motion-geodesic-jump-diffusion`) specify a coupled jump-diffusion SDE
({ref}`sec-the-coupled-jump-diffusion-sde`):

$$
dz_t = G^{-1} p_t\, dt, \qquad
dp_t = \bigl[-\nabla \Phi_{\text{eff}}(z) + u_\pi(z, a)\bigr]\, dt
       - \gamma\, p_t\, dt + \sigma\, dW_t
       + \Delta p \cdot dN_t
$$

where $G^{-1}$ is the inverse Poincare metric ({prf:ref}`def-mass-tensor`), $\Phi_{\text{eff}}$ is
the effective potential ({prf:ref}`def-effective-potential`), $u_\pi$ is the control field
({prf:ref}`def-the-control-field`), $\gamma$ is friction, $\sigma$ is noise strength tied to
cognitive temperature ({prf:ref}`def-cognitive-temperature`), and $dN_t$ is a Poisson jump process
({prf:ref}`prop-jump-intensity-from-value-discontinuity`).

:::{admonition} Researcher Bridge: World Models in RL
:class: info
Dreamer-v3 and PlaNet learn world models as recurrent neural networks predicting latent state
sequences. IRIS operates on discrete tokens from a VQ-VAE. Our approach is closest in spirit to
Hamiltonian Neural Networks (Greydanus et al., 2019) and Lagrangian Neural Networks (Cranmer et
al., 2020), but with three key differences: (1) we operate on a Riemannian manifold rather than
flat space, (2) the integrator is a symplectic splitting scheme with thermostat rather than
generic ODE integration, and (3) the WFR decomposition provides a principled mechanism for
discrete chart transitions that has no analogue in standard HNN/LNN work.
:::


(sec-world-model-architecture)=
## 2. Architecture Overview

:::{div} feynman-prose
Before we dive into any single piece, let me show you the whole machine. The world model is a
collection of modules, each responsible for one physical quantity. They share a common pattern:
a CovariantAttention block that reads chart tokens and produces a geometric output. This
uniformity is not an accident --- it ensures that every force component respects the same atlas
geometry that the encoder learned in Phase 1.

The key insight is *modularity*. The potential net knows nothing about the control field. The
curl net knows nothing about the jump process. Each module computes one thing well, and the
BAOAB integrator composes them into a complete dynamics step. This makes the system easy to
reason about, easy to debug, and easy to extend.
:::

```{mermaid}
flowchart TB
    subgraph Inputs["Inputs"]
        Z0["z₀ [B, D]<br/>initial position"]
        A["actions [B, H, A]<br/>action sequence"]
        RW0["rw₀ [B, K]<br/>initial chart weights"]
    end

    subgraph Sub["Covariant Sub-Modules"]
        PN["CovariantPotentialNet<br/>force + Φ_eff"]
        CF["CovariantControlField<br/>u_π(z, a, rw)"]
        VC["CovariantValueCurl<br/>F_ij antisymmetric"]
        CT["CovariantChartTarget<br/>chart logits"]
        JR["CovariantJumpRate<br/>λ(z, K) ≥ 0"]
        MI["CovariantMomentumInit<br/>p = λ²·net(z)"]
        HD["HodgeDecomposer<br/>(diagnostic, no params)"]
    end

    subgraph Integrator["BAOAB Integration Loop"]
        CJ["Conditional Jump<br/>(Fisher-Rao)"]
        BS["BAOAB Sub-Steps<br/>(W₂ Transport)"]
        CJ --> BS
    end

    Z0 --> MI
    MI --> |"p₀"| CJ
    Z0 --> CJ
    RW0 --> CJ
    A --> CJ
    CT --> CJ

    CJ --> BS
    PN --> BS
    CF --> BS
    VC --> |"Boris rotation"| BS
    HD --> |"diagnostics"| BS

    BS --> OUT["z_trajectory [B, H, D]<br/>chart_logits [B, H, K]<br/>momenta [B, H, D]<br/>phi_eff [B, H, 1]<br/>energy_var (scalar)"]
```

The six learnable sub-modules of `GeometricWorldModel`:

:::{div} feynman-added
| Module | Input | Output | Role |
|--------|-------|--------|------|
| `CovariantPotentialNet` | $z$, router weights | force $F$ [B,D], potential $\Phi$ [B,1] | Conservative + risk forces |
| `CovariantControlField` | $z$, action, router weights | $u_\pi$ [B,D] | Policy-conditioned control force |
| `CovariantValueCurl` | $z$, action | $\mathcal{F}_{ij}$ [B,D,D] | Antisymmetric field for Boris rotation |
| `CovariantChartTarget` | $z$, action, router weights | logits [B,K] | Supervised chart transition prediction |
| `CovariantJumpRate` | $z$, router weights | $\lambda$ [B,1] | Poisson jump rate ($\geq 0$ via softplus) |
| `CovariantMomentumInit` | $z$ | $p_0$ [B,D] | Initial momentum $p = \lambda^2 \cdot \text{net}(z)$ |
:::

Plus one parameter-free diagnostic module:

- `HodgeDecomposer`: Decomposes total force into conservative + solenoidal + harmonic components,
  providing the Hodge ratios used by the consistency loss.


(sec-covariant-force-field)=
## 3. The Covariant Force Field

:::{div} feynman-prose
The force field is the heart of the world model. Everything else --- the integrator, the jumps,
the training --- exists to serve this one question: given the agent's current position $z$ in
the Poincare ball and its current chart context, what force should be applied?

The answer comes in three parts, and understanding why there are three parts is understanding
the whole design.

First, there is an *analytic* hyperbolic drive. This is not learned. It is a fixed function
of position that pushes particles toward the center of the ball. Think of it as gravity in
the latent space. Points near the boundary of the ball are "extreme" --- they have high
information content, high specificity --- and the drive pulls them back toward more moderate
positions. This regularizes the dynamics: without it, the learned forces could push particles
to the boundary where the metric blows up.

Second, there is a *learned critic force* $f_{\text{critic}}$. This is where the value function
lives. The critic attends over chart tokens using CovariantAttention, producing both a force
vector and a scalar potential $V$. The force tells the particle where to go; the potential tells
the energy conservation loss whether the particle is going there correctly.

Third, there is a *learned risk force* $f_{\text{risk}}$. This accounts for uncertainty ---
regions of state space where the model is less confident deserve a different force profile. The
risk head has the same architecture as the critic head but serves a different purpose: it
modulates behavior in uncertain regions, as described in
{ref}`sec-radial-generation-entropic-drift-and-policy-control`.

The three forces combine with learnable mixing coefficients:

$$
F = \alpha\, \nabla U(z)
  + (1-\alpha)\, f_{\text{critic}}(z, K)
  + \gamma_{\text{risk}}\, f_{\text{risk}}(z, K)
$$

where $\alpha$ (default 0.5) balances analytic drive against learned critic, and
$\gamma_{\text{risk}}$ (default 0.01) keeps the risk correction small.
:::

### The Analytic Hyperbolic Drive

The analytic potential and its gradient are computed without any learned parameters or autograd:

$$
U(z) = -2\operatorname{artanh}(\|z\|), \qquad
\frac{\partial U}{\partial z_i} = \frac{-2\, z_i}{\|z\| \,(1 - \|z\|^2)}
$$

```python
# CovariantPotentialNet._analytic_U_and_grad (covariant_world_model.py)
r = z.norm(dim=-1, keepdim=True).clamp(min=1e-6, max=1.0 - 1e-6)
U = -2.0 * torch.atanh(r)                      # [B, 1]
dU_dz = -2.0 * z / (r * (1.0 - r ** 2))        # [B, D]
```

:::{note}
:class: feynman-added
The clamping at both ends is essential. At the origin ($\|z\| = 0$), the gradient would
contain a division by zero. At the boundary ($\|z\| = 1$), both $\operatorname{artanh}$ and
the denominator $(1 - \|z\|^2)$ diverge. The clamps at $10^{-6}$ and $1 - 10^{-6}$ keep the
numerics stable without affecting the physics in the interior of the ball.
:::

### Learned Critic and Risk Forces

Both the critic and risk heads follow an identical pattern: CovariantAttention over chart
tokens, then linear projections to a force vector and a scalar potential.

```python
# CovariantPotentialNet.force_and_potential (covariant_world_model.py)
ctx_x, ctx_z = self.chart_tok(rw, z)           # chart context tokens
x_q = self.z_embed(z)                           # query embedding

# Critic head: one CovariantAttention → force + scalar
v_feat, _ = self.v_critic_attn(z, ctx_z, x_q, ctx_x, ctx_x)
f_critic = self.v_force_out(v_feat)              # [B, D]
V = self.v_out(v_feat)                           # [B, 1]

# Risk head: same pattern, separate parameters
psi_feat, _ = self.psi_risk_attn(z, ctx_z, x_q, ctx_x, ctx_x)
f_risk = self.psi_force_out(psi_feat)            # [B, D]
Psi = self.psi_out(psi_feat)                     # [B, 1]
```

The effective potential assembles the scalar components for the energy conservation diagnostic:

$$
\Phi_{\text{eff}} = \alpha\, U(z)
  + (1 - \alpha)\, V_{\text{critic}}(z, K)
  + \gamma_{\text{risk}}\, \Psi_{\text{risk}}(z, K)
$$

This matches the definition in {prf:ref}`def-effective-potential`. The scalar $\Phi_{\text{eff}}$
is not used to compute forces (those come from the direct force predictions), but it provides
the potential energy for the Hamiltonian $H = \Phi_{\text{eff}} + \frac{1}{2} p^T G^{-1} p$
that the energy conservation loss monitors.


(sec-control-field-impl)=
## 4. The Control Field --- Action-Conditioned Forces

:::{div} feynman-prose
Now we come to the part that makes this a *control* system rather than just a simulator.
The control field $u_\pi$ ({prf:ref}`def-the-control-field`) translates the agent's action
into a force on the latent state. Without it, the world model would evolve autonomously ---
particles would roll downhill under the potential and diffuse under the thermostat. The
control field is how the agent steers.

The implementation follows the same CovariantAttention pattern as the potential net, but with
a crucial difference: the context tokens include both *action tokens* and *chart tokens*. The
action tokenizer lifts the flat action vector $a \in \mathbb{R}^A$ into $A$ separate tokens,
each carrying information about one action dimension. The chart tokenizer provides the atlas
context. CovariantAttention composes these into a single force vector that respects the
geometry and conditions on both the action and the chart structure.

Why tokenize the action? Because CovariantAttention operates on *sets of context vectors at
positions in the Poincare ball*. A flat action vector has no position. The tokenizer gives
each action dimension a position (the current $z$) and a feature embedding, making it
compatible with the attention mechanism. See {ref}`sec-policy-control-field` for the
theoretical motivation.
:::

### The Tokenizer Pattern

Both `ActionTokenizer` and `ChartTokenizer` produce pairs of tensors:
`(tokens_x, tokens_z)` --- feature tokens and position tokens.

```python
# ActionTokenizer.forward (covariant_world_model.py)
tokens_x = (action.unsqueeze(-1) * self.weight.unsqueeze(0)
            + self.pos_embed.unsqueeze(0))                                  # [B, A, d_model]
tokens_z = z.unsqueeze(1).expand(-1, self.action_dim, -1).contiguous()      # [B, A, D]
```

```python
# ChartTokenizer.forward (covariant_world_model.py)
tokens_x = rw.unsqueeze(-1) * self.chart_embeddings.unsqueeze(0)            # [B, K, d_model]
safe_centers = _project_to_ball(self.chart_centers)                          # [K, D]
tokens_z = safe_centers.unsqueeze(0).expand(B, -1, -1).contiguous()          # [B, K, D]
```

The `CovariantControlField` concatenates action and chart tokens into a single context set:

```python
# CovariantControlField.forward (covariant_world_model.py)
act_x, act_z = self.action_tok(action, z)
chart_x, chart_z = self.chart_tok(rw, z)
ctx_x = torch.cat([act_x, chart_x], dim=1)   # [B, A+K, d_model]
ctx_z = torch.cat([act_z, chart_z], dim=1)    # [B, A+K, D]
x_q = self.z_embed(z)
output, _ = self.attn(z, ctx_z, x_q, ctx_x, ctx_x)
return self.out(output)                        # [B, D] control force
```

:::{note}
:class: feynman-added
The chart tokenizer positions are the *chart centers themselves*, not the current position $z$.
This means the CovariantAttention mechanism computes hyperbolic distances between $z$ and each
chart center, giving the attention weights a geometric interpretation: the force contribution
from chart $k$ is weighted by how close $z$ is to chart $k$'s center. This is the attention
analogue of a partition-of-unity on the atlas.
:::


(sec-boris-baoab)=
## 5. The Geodesic Boris-BAOAB Integrator

:::{div} feynman-prose
Here is where the physics happens. The BAOAB integrator
({prf:ref}`def-baoab-splitting`) is a particular way of splitting the Hamiltonian
dynamics into steps that can each be handled exactly or semi-exactly. The name tells you
the order: **B**-step (momentum kick from forces), **A**-step (position drift), **O**-step
(thermostat noise), **A**-step (position drift again), **B**-step (second force evaluation).

Why this particular splitting? Because it has remarkable properties. The O-step in the middle
acts as a thermostat --- it injects noise and damps momentum, maintaining the system at the
cognitive temperature $T_c$. The symmetric placement of the A-steps and B-steps around the
O-step gives second-order accuracy in the Hamiltonian limit ($T_c \to 0$). And crucially, the
BAOAB scheme preserves the Boltzmann distribution
({prf:ref}`prop-baoab-preserves-boltzmann`), meaning that in equilibrium the sampled states
have the correct thermodynamic weights.

But our BAOAB is not the textbook version. We are on a curved space, so the A-steps must use
the Poincare exponential map. We add Christoffel corrections to account for geodesic drift.
We add Boris rotation for curl forces that preserve the momentum norm. And we add CFL bounds
to prevent numerical explosions near the boundary.

Let me walk you through one complete step.
:::

### B Step (First Half): Momentum Kick

The B-step applies forces to the momentum. Hamilton's equation says $dp/dt = -F + u_\pi$,
where $F$ is the potential force and $u_\pi$ is the control force. For a half-step:

```python
# GeometricWorldModel._baoab_step — B step (covariant_world_model.py)
force, _ = self.potential_net.force_and_potential(z, rw)   # [B, D]
u_pi = self.control_net(z, action, rw)                     # [B, D]
kick = force - u_pi                                         # [B, D] net force
```

When CFL bounds are active (`min_length > 0`), the force is squashed via a smooth,
direction-preserving map:

$$
\psi_F(\text{kick}) = \frac{F_{\max} \cdot \text{kick}}{F_{\max} + \|\text{kick}\|}, \qquad
F_{\max} = \frac{2\gamma \,\ell_{\min}}{\Delta t}
$$

```python
if self.F_max > 0:
    kick = self.F_max * kick / (self.F_max + kick.norm(dim=-1, keepdim=True))
```

The B-step uses a **Strang splitting** to handle the non-commuting conservative and
solenoidal forces at second-order accuracy:

$$
B = B_{\text{cons}}(\tfrac{h}{2}) \;\circ\; B_{\text{sol}}(h) \;\circ\; B_{\text{cons}}(\tfrac{h}{2})
$$

```python
# B step — Strang splitting (covariant_world_model.py, lines 781-792)
p_minus = p - (h / 2.0) * kick                                    # conservative half-kick
p_plus, _ = self._boris_rotation(p_minus, z, action, curl_F=curl_F)  # solenoidal (Boris)
# ... Hodge decomposition (diagnostic) ...
p = p_plus - (h / 2.0) * kick                                     # conservative half-kick
```

The same `kick` vector is applied twice with coefficient $h/2$, sandwiching the Boris rotation.
This symmetric splitting ensures second-order accuracy in both force components even though the
conservative gradient $\nabla\Phi$ and solenoidal curl $\mathcal{F}_{ij}$ do not commute.

### Boris Rotation: Norm-Preserving Curl Forces

The Boris rotation ({prf:ref}`def-baoab-splitting`) handles the antisymmetric (curl) component
of the force field. In plasma physics this is the Lorentz force from a magnetic field; here it
is the value-curl field $\mathcal{F}_{ij}$ predicted by `CovariantValueCurl`.

The algorithm ensures $\|p_{\text{plus}}\| = \|p_{\text{minus}}\|$ exactly, regardless of the
field strength:

```python
# GeometricWorldModel._boris_rotation (covariant_world_model.py)
T = (h / 2.0) * self.beta_curl * lambda_inv_sq.unsqueeze(-1) * F   # [B, D, D]

# Half-rotation
t_vec = torch.bmm(T, p_minus.unsqueeze(-1)).squeeze(-1)            # [B, D]
p_prime = p_minus + t_vec

# Full-rotation correction
t_sq = (T ** 2).sum(dim=(-2, -1))                                  # [B]
s_factor = 2.0 / (1.0 + t_sq).unsqueeze(-1)                        # [B, 1]
s_vec = s_factor * torch.bmm(T, p_prime.unsqueeze(-1)).squeeze(-1)  # [B, D]

p_plus = p_minus + s_vec  # |p_plus| = |p_minus| exactly
```

:::{div} feynman-prose
The Boris algorithm is a beautiful piece of numerical engineering. The antisymmetric matrix $T$
rotates the momentum without changing its magnitude --- that is a property of antisymmetric
matrices. But a naive rotation $p + T \cdot p$ is only first-order accurate. The Boris trick
uses two half-steps with a correction factor $2/(1 + \|T\|_F^2)$ that makes the rotation
exact to machine precision, regardless of the step size or field strength. This is why plasma
physicists love it, and why we use it here for the curl forces from the value field
$\mathcal{F}_{ij}$.

The `CovariantValueCurl` module predicts the upper triangle of the antisymmetric tensor
$\mathcal{F}_{ij}$ from the position and action via CovariantAttention. The lower triangle
is filled by antisymmetry: $\mathcal{F}_{ji} = -\mathcal{F}_{ij}$. This guarantees the
norm-preservation property by construction.
:::

### A Step: Geodesic Drift with Christoffel Correction

The A-step advances the position along the momentum direction. On a curved space, this is
not just $z \leftarrow z + h \cdot v$. We must use the Poincare exponential map and correct
for geodesic curvature via the Christoffel symbols
({prf:ref}`prop-explicit-christoffel-symbols-for-poincare-disk`):

```python
# A step (first half) — covariant_world_model.py
cf = self.metric.conformal_factor(z)                        # [B, 1]
lambda_inv_sq = 1.0 / (cf ** 2 + epsilon)                   # [B, 1]
v = lambda_inv_sq * p                                        # [B, D] contravariant velocity
geo_corr = christoffel_contraction(z, v)                     # [B, D] Γ^i_{jk} v^j v^k
v_corr = v - (h / 4.0) * geo_corr                           # corrected velocity
```

The velocity is squashed via the CFL bound $V_{\text{alg}} = \ell_{\min} / \Delta t$:

```python
if self.V_alg > 0:
    v_corr = self.V_alg * v_corr / (self.V_alg + v_corr.norm(dim=-1, keepdim=True))
```

Then the position is updated via the exponential map and projected back into the ball:

```python
z = poincare_exp_map(z, (h / 2.0) * v_corr)                # geodesic step
z = _project_to_ball(z)                                      # safety clamp |z| < 0.99
```

:::{note}
:class: feynman-added
The Christoffel correction factor is $h/4$ rather than $h/2$ because the A-step is itself
a half-step ($h/2$ drift). The correction at quarter-step gives a centered approximation
to the geodesic equation $\ddot{z}^i + \Gamma^i_{jk} \dot{z}^j \dot{z}^k = 0$ within the
half-step.
:::

### O Step: Ornstein-Uhlenbeck Thermostat

The O-step couples the system to a heat bath at cognitive temperature $T_c$
({prf:ref}`def-cognitive-temperature`):

$$
p \leftarrow c_1 \cdot p + c_2 \cdot \lambda(z) \cdot \xi, \qquad
c_1 = e^{-\gamma \Delta t}, \quad
c_2 = \sqrt{(1 - c_1^2)\, T_c}, \quad
\xi \sim \mathcal{N}(0, I)
$$

```python
# O step — covariant_world_model.py
cf = self.metric.conformal_factor(z)           # [B, 1]
if self.cf_max > 0:
    cf = self.cf_max * torch.tanh(cf / self.cf_max)   # smooth cap
xi = torch.randn_like(p)
p = self.c1 * p + self.c2 * cf * xi
```

:::{div} feynman-prose
The conformal factor $\lambda(z) = 2/(1 - \|z\|^2)$ in the noise term is essential. Near the
boundary, $\lambda$ diverges. Without a cap, the thermostat would inject enormous noise at
boundary points, destabilizing the integration. The smooth tanh cap ensures $\lambda$ never
exceeds $\lambda_{\max}$, which is derived from the CFL condition:

$$
\lambda_{\max} = \frac{2\, \ell_{\min}}{c_2 \sqrt{D}\, \Delta t}
$$

This is the third and final CFL bound (alongside $F_{\max}$ and $V_{\text{alg}}$), and all
three come from the same minimum length scale $\ell_{\min}$. One number controls all the
stability bounds. That is the payoff of deriving everything from the geometry of the
Poincare metric ({ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits`).
:::

### Second A and B Steps

The BAOAB scheme is symmetric: after the O-step, we repeat the A-step (second-half drift)
and B-step (second-half kick with a *new* force evaluation at the updated position). The
second force evaluation is what makes BAOAB velocity-Verlet-like and second-order accurate.

```python
# B step (second half) — second force evaluation at new z
force2, phi_eff = self.potential_net.force_and_potential(z, rw)
u_pi2 = self.control_net(z, action, rw)
kick2 = force2 - u_pi2
# ... same CFL squashing ...
p = p - (h / 2.0) * kick2
```

### The Hodge Decomposition Diagnostic

After each BAOAB step, the `HodgeDecomposer` splits the total non-control force into three
orthogonal components:

$$
f_{\text{total}} = f_{\text{conservative}} + f_{\text{solenoidal}} + f_{\text{harmonic}}
$$

where $f_{\text{conservative}}$ comes from the potential net, $f_{\text{solenoidal}}$ comes
from the Boris rotation impulse, and $f_{\text{harmonic}}$ is the residual. The Hodge
consistency loss penalizes $\|f_{\text{harmonic}}\|^2$ --- a well-structured model should
explain all forces through either the gradient field or the curl field.

### CFL Stability Summary

:::{div} feynman-added
| Bound | Formula | Applied To | Method |
|-------|---------|------------|--------|
| Force | $F_{\max} = 2\gamma\, \ell_{\min} / \Delta t$ | `kick` in B-step | Smooth squashing $\psi_F$ |
| Velocity | $V_{\text{alg}} = \ell_{\min} / \Delta t$ | `v_corr` in A-step | Smooth squashing $\psi_v$ |
| Conformal factor | $\lambda_{\max} = 2\ell_{\min} / (c_2 \sqrt{D}\, \Delta t)$ | $\lambda$ in O-step | Tanh cap |
:::

All three bounds are derived from one parameter: `min_length` ($\ell_{\min} = 0.03$ by default).
When `min_length = 0`, all squashing is disabled for backward compatibility.

:::{admonition} Researcher Bridge: BAOAB and Symplectic Integrators
:class: info
The BAOAB splitting was introduced by Leimkuhler and Matthews (2013) for Langevin dynamics in
molecular simulation, where it outperforms ABOBA and BABO splittings for configurational
sampling. The Boris rotation algorithm originates from particle-in-cell plasma simulations
(Boris 1970) and is widely used in electromagnetic particle tracking because of its exact
norm-preservation property. Our contribution is combining these two techniques on a Riemannian
manifold with Christoffel corrections and deriving all CFL bounds from a single geometric
length scale, rather than tuning stability parameters independently. The closest related work
in machine learning is Hamiltonian Monte Carlo on manifolds (Girolami and Calderhead, 2011),
but that work uses geodesic integration for sampling rather than for learned dynamics prediction.
:::

:::{admonition} Researcher Bridge: Screened Poisson Critic
:class: info
The screened Poisson PDE loss connects to the literature on physics-informed neural networks
(PINNs) by Raissi, Perdikaris, and Karniadakis (2019). However, our usage is specific to the
hyperbolic setting: the Laplace-Beltrami operator $\Delta_G$ on the Poincare ball includes a
conformal correction term that has no Euclidean analogue. The Hutchinson trace estimator for
the Hessian (Hutchinson 1990) avoids second-order autograd, keeping the computational cost
linear in the latent dimension. The screening mass $\kappa$ controls the decay length of the
value function's response to reward signals, connecting to the discount factor $\gamma$ in
standard RL via $\kappa^2 \sim 1/(1-\gamma)$.
:::


(sec-conditional-jump)=
## 6. Conditional Chart Jumps --- The Fisher-Rao Component

:::{div} feynman-prose
So far everything has been smooth flow within a chart. But the WFR geometry
({ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`) tells us
that state evolution also includes *discrete jumps* between charts. This is the Fisher-Rao
component: it handles the "reaction" part of the transport-reaction decomposition.

The idea is simple. At each horizon step, before the BAOAB sub-steps, the world model asks:
"Should this particle stay in its current chart, or jump to a different one?" If the answer is
"jump," the particle *teleports* to the target chart center and its momentum is reinitialized.
No smooth trajectory connects the old and new positions --- that is what makes it a jump, not a
drift.

The decision is driven by Boltzmann logits that balance value gain against transport cost. A
particle jumps to chart $k$ if the value at chart center $c_k$ is sufficiently better than
the current value, minus the geodesic cost of getting there:

$$
\text{logit}_k = \beta \bigl[V(c_k) - V(z) - d_H(z, c_k)\bigr]
$$

Higher $\beta$ makes jumps rarer but more decisive. Lower $\beta$ makes the system more
exploratory. This connects to the cognitive temperature and the exploration-exploitation
tradeoff discussed in {ref}`sec-belief-dynamics-prediction-update-projection`.
:::

### The Jump Decision

The `_conditional_jump` method computes two sets of chart logits:

1. **Boltzmann logits** (detached, no gradient): used for the actual jump *decision*.
2. **Supervised chart logits** (gradient-carrying): from `CovariantChartTarget`, used for the
   chart transition CE loss.

```python
# GeometricWorldModel._conditional_jump (covariant_world_model.py)
chart_logits = self.chart_predictor(z, action, rw)         # supervised (has grad)
boltz_logits = self._boltzmann_chart_logits(z, rw)          # detached (no grad)

current_chart = rw.argmax(dim=-1)                           # [B]
target_chart = boltz_logits.argmax(dim=-1)                   # [B]
jumped = current_chart != target_chart                       # [B]
```

:::{div} feynman-prose
Why two sets of logits? This is subtle and worth understanding.

The Boltzmann logits involve evaluating the potential $V(c_k)$ at every chart center, which is
expensive and creates a large computation graph. If we backpropagated through these logits, the
Phase 3 `retain_graph=True` calls would blow up GPU memory. So the Boltzmann logits are
detached: they drive the jump *decision* but carry no gradient.

The supervised chart logits from `CovariantChartTarget` are cheap (one attention pass) and
carry gradients. The chart transition CE loss trains this module to *predict* which chart the
Boltzmann process would choose, without needing to evaluate the potential at every center
during training. At inference time, the model can use either pathway.
:::

### Teleportation and Momentum Reset

When a jump occurs, the particle teleports to the target chart center and its momentum is
reinitialized:

```python
if jumped.any():
    centers = self.chart_predictor.chart_tok.chart_centers     # [K, D]
    target_centers = centers[target_chart]                      # [B, D]

    z_jumped = _project_to_ball(target_centers)
    z = torch.where(jumped.unsqueeze(-1), z_jumped, z)

    p_new = self.momentum_init(z_jumped)                        # fresh momentum
    p = torch.where(jumped.unsqueeze(-1), p_new, p)
```

After the jump, the router weights are updated from the supervised chart logits (not the
Boltzmann logits), ensuring gradients flow through the chart prediction pathway:

```python
rw = F.softmax(chart_logits, dim=-1)                           # gradient-carrying
```

### The Forward Rollout

The complete forward pass loops over the horizon $H$, applying conditional jumps then BAOAB
sub-steps at each step:

```python
# GeometricWorldModel.forward (covariant_world_model.py)
p = self.momentum_init(z_0)                         # [B, D]
z = z_0; rw = router_weights_0

for t in range(H):
    action_t = actions[:, t, :]                     # [B, A]

    # Step 1: Conditional jump (Fisher-Rao)
    if self.use_jump:
        z, p, cl, rw, jumped = self._conditional_jump(z, p, action_t, rw)

    # Steps 2..N+1: BAOAB sub-steps (W₂ transport)
    for s in range(N):
        z, p, phi_eff, hodge_info = self._baoab_step(z, p, action_t, rw)
        # Track Hamiltonian H = Φ_eff + ½ g^{-1} |p|²
        H_s = phi_eff + 0.5 * g_inv * p_sq
        energy_substeps[:, t, s] = H_s.squeeze(-1)
```

The energy variance across sub-steps is returned for the conservation loss:

```python
energy_var = energy_substeps.var(dim=-1).mean()
```


(sec-world-model-losses)=
## 7. Loss Functions --- Teaching the World Model Physics

:::{div} feynman-prose
A world model that can integrate Hamiltonian dynamics is useless if the forces are wrong. The
loss functions teach the network *which* forces to predict. There are six dynamics losses, each
enforcing a different physical constraint, plus a supervised geodesic diffusion mode that
provides stronger waypoint-level supervision.

Let me tell you what each loss does and why it is there. I want you to see them not as a grab
bag of regularizers but as a coherent set of requirements that together define what it means
for the world model to be correct.
:::

### Dynamics Losses

**Geodesic loss.** The primary training signal: how far are the predicted positions from the
targets?

$$
\mathcal{L}_{\text{geo}} = \frac{1}{B \cdot H} \sum_{b,t} d_H(z^{\text{pred}}_{b,t},\; z^{\text{target}}_{b,t})
$$

```python
# compute_dynamics_geodesic_loss (losses.py)
pred_flat = z_pred.reshape(B * H, D)
tgt_flat = z_target.reshape(B * H, D)
return hyperbolic_distance(pred_flat, tgt_flat).mean()
```

**Chart transition loss.** Cross-entropy between predicted chart logits and ground-truth chart
assignments. This trains the `CovariantChartTarget` module.

$$
\mathcal{L}_{\text{chart}} = \mathrm{CE}(\text{chart\_logits}_{b,t},\; K^{\text{target}}_{b,t})
$$

**Momentum regularization.** Prevents runaway momenta by penalizing kinetic energy, measured
with the inverse metric to respect the geometry:

$$
\mathcal{L}_{\text{mom}} = \frac{1}{2} \left\langle g^{-1}(z)\, \|p\|^2 \right\rangle, \qquad
g^{-1}(z) = \left(\frac{1 - \|z\|^2}{2}\right)^2
$$

```python
# compute_momentum_regularization (losses.py)
r_sq = (z_trajectory ** 2).sum(dim=-1, keepdim=True)       # [B, H, 1]
g_inv_factor = ((1.0 - r_sq).clamp(min=1e-6) / 2.0) ** 2  # [B, H, 1]
p_sq = (momenta ** 2).sum(dim=-1, keepdim=True)             # [B, H, 1]
kinetic = 0.5 * g_inv_factor * p_sq
return kinetic.mean()
```

**Energy conservation loss.** A symplectic integrator should approximately conserve the
Hamiltonian $H = \Phi_{\text{eff}} + \frac{1}{2} p^T G^{-1} p$. We penalize the variance
of $H$ across BAOAB sub-steps:

$$
\mathcal{L}_{\text{energy}} = \frac{1}{B} \sum_b \operatorname{Var}_{s}
  \bigl[H(z_{b,s}, p_{b,s})\bigr]
$$

:::{div} feynman-prose
This is genuinely subtle. The BAOAB integrator is *not* exactly symplectic --- the thermostat
breaks time-reversal symmetry. So the Hamiltonian is not exactly conserved. But it should be
*approximately* conserved between O-steps: the B and A steps are symplectic, and only the O-step
injects noise. If $H$ varies wildly, the forces are producing unphysical trajectories.

There are two granularities of energy monitoring in the code. The `compute_energy_conservation_loss`
function in `losses.py` measures the variance of $H$ across **horizon steps** (the $t$-axis).
Meanwhile, the `forward` method in `GeometricWorldModel` tracks `energy_substeps[:, t, s]` across
the $N$ BAOAB sub-steps within each horizon step and reports `energy_substeps.var(dim=-1).mean()`
--- the **sub-step variance**. The sub-step variance is the tighter diagnostic: it checks whether
a single integration step conserves energy, before thermostat noise accumulates over the horizon.
:::

**Hodge consistency loss.** Penalizes the harmonic residual in the Hodge decomposition:

$$
\mathcal{L}_{\text{Hodge}} = \|f_{\text{harmonic}}\|^2
$$

A well-structured model should explain all forces through either the conservative potential or
the solenoidal curl field. The harmonic component is a residual that should be small.

**Screened Poisson PDE loss.** Enforces that the learned critic $V$ approximately satisfies
the screened Poisson equation on the Poincare ball:

$$
\mathcal{L}_{\text{PDE}} = \left\|(-\Delta_G + \kappa^2)\, V - \rho_r\right\|^2
$$

where $\Delta_G$ is the Laplace-Beltrami operator, $\kappa$ is a screening mass, and
$\rho_r = d_H(z, z_{\text{target}})$ is a reward-density proxy.

The hyperbolic Laplacian is computed via Hutchinson trace estimation with finite differences
--- no second-order autograd required:

$$
\Delta_G V = \lambda^{-2}\bigl[\Delta_E V + (D-2)\,\lambda\,(z \cdot \nabla V)\bigr]
$$

```python
# hyperbolic_laplacian (losses.py) — Hutchinson trace + radial correction
probes = torch.randint(0, 2, (n_probes, N, D), ...) * 2 - 1  # Rademacher
# ... single batched V_func call for all perturbation points ...
laplacian_E = trace_terms.sum(dim=0) / (n_probes * eps ** 2)   # [N, 1]
z_dot_grad = z_norm * (V_zhat_plus - V_zhat_minus) / (2 * eps) # [N, 1]
lap_G = inv_lambda_sq * (laplacian_E + (D - 2) * lambda_z * z_dot_grad)
```

### Supervised Geodesic Diffusion

For Phase 2 training, there is an alternative to the standard rollout loss: **supervised
geodesic diffusion**. Instead of rolling out the world model and comparing endpoints, we
create intermediate waypoints along the geodesic between consecutive encoded positions and
compare the world model's integration against these waypoints.

**Step 1. Geodesic interpolation.** Create $N+1$ waypoints along the Poincare geodesic from
$z_t$ to $z_{t+1}$:

$$
v = \log_{z_t}(z_{t+1}), \qquad z_k = \exp_{z_t}\!\left(\frac{k}{N} \cdot v\right)
$$

**Step 2. Momentum targets.** Finite-difference momenta from the waypoints:

$$
p_k = \frac{\log_{z_k}(z_{k+1})}{\Delta t}
$$

**Step 3. Supervised integration.** Run the world model deterministically (O-step noise zeroed)
from $z_0$ with initial momentum $p_0$, then compare against the waypoint trajectory.

The supervised loss combines three terms:

$$
\mathcal{L}_{\text{diffusion}} = w_{\text{pos}}\, \mathcal{L}_{\text{pos}}
  + w_{\text{end}}\, \mathcal{L}_{\text{end}}
  + w_{\text{mom}}\, \mathcal{L}_{\text{mom}}
$$

where $\mathcal{L}_{\text{pos}}$ is the mean geodesic distance across all waypoints,
$\mathcal{L}_{\text{end}}$ is the endpoint distance, and $\mathcal{L}_{\text{mom}}$ is
the metric-aware momentum error $g^{-1}\|p_{\text{pred}} - p_{\text{target}}\|^2$.

:::{div} feynman-prose
The geodesic diffusion mode is more expensive than the standard rollout loss (it does $N$
BAOAB steps per pair instead of $N$ steps for the whole horizon) but provides much richer
supervision. Every intermediate waypoint gives a gradient signal, not just the endpoint.
This is especially helpful early in Phase 2 when the world model has no idea what forces to
produce --- the waypoint supervision tells it exactly where to go at every sub-step.

For pairs that cross chart boundaries, only the chart transition CE is computed (no supervised
integration). The jump operator handles cross-chart dynamics; the BAOAB integrator handles
within-chart dynamics. Each is trained on the cases where it applies.
:::

### Phase Loss Assemblers

The loss functions are composed by phase-specific assemblers:

**`compute_phase2_loss`**: Assembles rollout-mode losses.

$$
\mathcal{L}_2 = w_{\text{geo}} \mathcal{L}_{\text{geo}}
  + w_{\text{chart}} \mathcal{L}_{\text{chart}}
  + w_{\text{mom}} \mathcal{L}_{\text{mom}}
  + w_{\text{energy}} \mathcal{L}_{\text{energy}}
  + w_{\text{Hodge}} \mathcal{L}_{\text{Hodge}}
  + w_{\text{PDE}} \mathcal{L}_{\text{PDE}}
$$

**`compute_phase2_geodesic_diffusion_loss`**: Loops over consecutive pairs, applies supervised
integration for same-chart pairs and chart CE for cross-chart pairs.

**`compute_phase3_loss`**: Combines scaled Phase 1 encoder losses with Phase 2 dynamics losses,
returning split components for gradient surgery.


(sec-three-phase-training)=
## 8. The Three-Phase Training Pipeline

:::{div} feynman-prose
Now I need to tell you about something that is more engineering than physics, but it is
essential engineering. You cannot train all of this at once from scratch. The encoder needs
to find a good atlas before the world model can learn forces on that atlas. And the world
model needs stable dynamics before you try to fine-tune the encoder through it. The three-phase
pipeline respects these dependencies.

Think of it like building a house. Phase 1: lay the foundation (the atlas and codebook).
Phase 2: build the walls (the dynamics model). Phase 3: the finishing work, where everything
has to fit together precisely. If you try to do the finishing work before the walls are up,
you get a mess.
:::

### Phase 1: Encoder Training

Phase 1 trains the encoder on single frames, as described in
{ref}`sec-shared-dynamics-encoder`. The world model does not exist yet. The loss is
`compute_phase1_loss` plus orthogonality and jump consistency terms. Three parameter groups
with differentiated learning rates:

:::{div} feynman-added
| Parameter Group | Learning Rate | Rationale |
|----------------|---------------|-----------|
| Base encoder | `lr_encoder` | Standard training rate |
| Chart centers | `lr_encoder * 0.1` | Centers should move slowly for atlas stability |
| Codebook entries | `lr_encoder * 0.5` | Codes track encoder but avoid oscillation |
:::

### Phase 2: World Model Training (Frozen Encoder)

Phase 2 freezes the entire encoder and trains only the world model. The key steps:

**1. Bind chart centers.** The Phase 1 chart centers are copied into every chart-conditioned
sub-module of the world model via `bind_chart_centers`:

```python
# _run_phase2 (train.py)
phase1_centers = getattr(inner_enc, "chart_centers", None)
world_model.bind_chart_centers(phase1_centers, freeze=True)
```

This ensures the world model operates on the same atlas geometry as the encoder.

**2. Disable jumps.** During Phase 2, the jump process is disabled
(`world_model.use_jump = False`). The chart prediction module is still trained via the chart
CE loss, but no actual teleportation occurs. This lets the BAOAB integrator learn stable
within-chart dynamics before jumps add discontinuities.

**3. Encode all frames.** All frames in each sequence are encoded under `torch.no_grad()`.
The encoded positions $z_{\text{all}}$, router weights, and chart indices become the
supervision targets.

**4. Two training modes.** If `use_geodesic_diffusion` is True, the loss is
`compute_phase2_geodesic_diffusion_loss` (supervised waypoint matching). Otherwise, it is
`compute_phase2_loss` (standard rollout + geodesic distance).

**5. Optional dynamics codebook.** If `dyn_codes_per_chart > 0`, a separate
`DynamicsTransitionModel` is trained alongside the world model, providing the macro-Markov
closure signal described in the encoder chapter.

### Phase 3: Joint Fine-Tuning

Phase 3 unfreezes the encoder and trains both systems jointly. This is the most delicate phase,
using **alternating optimization** with gradient surgery to prevent the encoder and world model
from destabilizing each other.

:::{div} feynman-prose
Here is where it gets interesting. In Phase 3, the encoder and world model are trained
simultaneously, but not in the same backward pass. They take turns.

First, the **encoder step**: the world model is frozen, and the encoder is updated using
Phase 1 losses (scaled down by `phase3_encoder_scale = 0.1`) plus the enclosure probe loss
and Zeno smoothness penalty. The small scale factor is crucial: we do not want the encoder
to forget everything it learned in Phase 1. We just want it to *refine* its representations
in light of what the world model needs.

Second, the **WM step**: the encoder outputs are detached (no gradient flows back to the
encoder), and the world model is updated using Phase 2 losses (scaled by
`phase3_dynamics_scale = 1.0`). The detachment is the gradient surgery: the world model
adapts to the encoder's current output, but does not try to change it.

Third, the **probe step**: the enclosure probe is updated on its own, with all inputs detached.
This keeps the probe's learning dynamics independent from the main optimization.

Why alternating? Because joint optimization with shared gradients is unstable. The encoder
changes its output, the world model's predictions become wrong, the loss spikes, gradients
explode, everything falls apart. Alternating optimization with detachment keeps each system
stable while allowing them to co-adapt gradually.
:::

The Phase 3 training step in detail:

```python
# _run_phase3 — encoder step (train.py)
L_enc = (
    config.phase3_encoder_scale * base_enc
    + config.phase3_zn_reg_scale * zn_reg
)
if L_encl_encoder is not None:                       # guard: probe may be disabled
    L_enc = L_enc + config.w_enclosure * L_encl_encoder
if config.w_zeno > 0 and H > 1:                      # guard: Zeno needs horizon > 1
    L_enc = L_enc + config.w_zeno * L_zeno
L_enc.backward()
optimizer_enc.step()

# _run_phase3 — WM step (encoder detached)
z_all_det = z_all.detach()                           # gradient surgery
rw_all_det = rw_all.detach()
if config.use_geodesic_diffusion:                    # branch: geodesic diffusion vs rollout
    dyn_loss, dyn_metrics = compute_phase2_geodesic_diffusion_loss(...)
else:
    wm_output = world_model(z_all_det[:, 0], actions, rw_0)
    dyn_loss, dyn_metrics = compute_phase2_loss(wm_output, z_targets, ...)
(config.phase3_dynamics_scale * dyn_loss).backward()
optimizer_wm.step()

# _run_phase3 — probe step (separate optimizer, if probe exists)
if probe is not None and L_encl_probe is not None:
    probe_optimizer.zero_grad()
    L_encl_probe.backward()
    probe_optimizer.step()
```

### The Enclosure Probe in Phase 3

The `EnclosureProbe` ({prf:ref}`def-closure-defect`) is an adversarial probe that tests whether
$z_{\text{tex}}$ leaks dynamics information. It uses a gradient reversal layer (GRL) so that
improving the probe's ability to predict next-state from $z_{\text{tex}}$ *pushes the encoder
to hide dynamics information from $z_{\text{tex}}$*.

Two probes with shared architecture:

- **Full probe**: receives chart embedding + code embedding + action + GRL($z_{\text{tex}}$)
- **Baseline probe**: receives chart embedding + code embedding + action only

The **closure defect** is the accuracy gap: $\Delta = \text{acc}_{\text{full}} - \text{acc}_{\text{base}}$.
If $\Delta > 0$, $z_{\text{tex}}$ carries dynamics information that should live in $z_n$ or $z_q$.
The GRL gradient pushes the structure filter to eliminate this leakage, enforcing the causal
enclosure condition ({prf:ref}`def-causal-enclosure`).

The GRL alpha follows a linear warmup schedule to prevent the adversarial signal from
destabilizing early training:

```python
# grl_alpha_schedule (losses.py)
alpha = max_alpha * step / warmup_steps  # linear ramp to max_alpha
```

### Option D: Codebook Dynamics Step

When a codebook dynamics optimizer is configured (`optimizer_cb is not None` and horizon $H > 1$),
Phase 3 includes a fourth optimization step that fine-tunes the codebook embeddings through the
world model. This keeps the atlas geometry consistent with the learned dynamics:

```python
# _run_phase3 — codebook dynamics step (train.py, lines 655-713)
z_coarse_0 = project_to_ball(mobius_add(c_bar[:, 0].detach(), zq_blended[:, 0]))
# Freeze WM, roll out from coarse initial state
wm_output = world_model(z_coarse_0, actions, rw_0)
L_cb_dyn = hyperbolic_distance(z_pred, z_targets).mean()

# Optional: dynamics transition model on VQ codes
if dyn_trans_model is not None:
    L_cb_extra = vq_dyn_loss + w_dyn_transition * trans_loss
    L_cb_extra.backward()

(w_codebook_dynamics * L_cb_dyn).backward()
optimizer_cb.step()
```

:::{div} feynman-prose
The codebook dynamics step solves a subtle chicken-and-egg problem. The world model integrates
trajectories in chart-local coordinates, so the chart centers (codebook embeddings) define the
coordinate system. But the encoder learned those embeddings in Phase 1 without any dynamics
awareness. In Phase 3, the codebook step rolls out the world model from a *coarse* initial state
(blended codebook center + quantized offset) and pushes the codebook to minimize the hyperbolic
distance between predicted and observed trajectories. The world model parameters are frozen during
this step --- only the codebook moves. This aligns the atlas geometry with the dynamics without
destabilizing the integration.
:::

(sec-world-model-diagnostics)=
## 9. Diagnostics --- Reading the Instruments

:::{div} feynman-prose
You have built a Hamiltonian world model on a curved space with Boris rotation, CFL bounds,
chart jumps, and six loss functions. How do you know if it is working?

The answer is the same as it always is: you watch the instruments. But you need to know *which*
instruments matter and *what the readings mean*. Let me tell you what to look for.
:::

### Hodge Ratios

The Hodge decomposition gives you three numbers at each step:

- **Conservative ratio** $\|f_{\text{cons}}\| / \|f_{\text{total}}\|$: What fraction of the
  force comes from the learned potential. Should be dominant (0.6--0.9).
- **Solenoidal ratio** $\|f_{\text{sol}}\| / \|f_{\text{total}}\|$: What fraction comes from
  the curl/Boris field. Should be moderate (0.1--0.3).
- **Harmonic ratio** $\|f_{\text{harm}}\| / \|f_{\text{total}}\|$: The unexplained residual.
  Should be small ($< 0.1$). If this is large, the model is producing forces that are neither
  gradient nor curl --- something is wrong with the force decomposition.

### Energy Conservation

The `energy_var` metric tracks Hamiltonian variance across BAOAB sub-steps. For a well-behaved
integrator, this should be small. If it grows during training, the forces are becoming too
violent for the step size, and you may need to increase `n_refine_steps` or decrease `wm_dt`.

### Chart Prediction Accuracy

The `chart_accuracy` metric from the geodesic diffusion loss measures how well
`CovariantChartTarget` predicts the next chart. It should climb above chance ($1/K$) early
in Phase 2 and reach high accuracy by the end. If it stagnates, the chart structure from
Phase 1 may not be dynamically informative.

### Geodesic Miss Distance

The `geo_miss` metric is the hyperbolic distance between the predicted and target final
positions after supervised integration. This is the most direct measure of world model
quality. Track it across epochs and across chart-pair types (same-chart vs. cross-chart).

:::{admonition} Training Health Checklist
:class: feynman-added tip

**Phase 2 early (epochs 1--10):**
- Geodesic loss should drop. If not: check that chart centers were bound correctly.
- Energy conservation should be moderate. If wild: reduce `wm_dt` or increase `wm_min_length`.
- Momentum regularization should be declining. If not: increase `w_momentum_reg`.

**Phase 2 late (epochs 20--50):**
- Chart accuracy should be well above chance. If stuck: check Phase 1 chart structure.
- Hodge harmonic ratio should be below 0.1. If not: increase `w_hodge`.
- Screened Poisson PDE residual should decrease. If not: check `wm_screening_kappa`.
- `geo_miss` should be small for same-chart pairs. If not: increase `wm_diffusion_substeps`.

**Phase 3 (epochs 1--30):**
- Encoder losses should not spike. If they do: reduce `phase3_encoder_scale`.
- Dynamics losses should stay at or improve on Phase 2 levels. If degrading: alternating
  optimization is unstable; reduce learning rates.
- Enclosure defect should decrease toward zero. If increasing: the structure filter is failing;
  check $z_n$/$z_{\text{tex}}$ norms and orthogonality loss.
- Zeno loss should remain low. If increasing: chart routing is flickering; check routing
  entropy and margin.

**Warning signs at any phase:**
- `NaN` in any loss: numerical instability. Check `min_length > 0` and gradient clipping.
- Energy variance growing epoch over epoch: forces are too large for the step size.
- Harmonic ratio near 1.0: the curl and potential nets are not learning; check learning rate.
- Momentum norms growing without bound: friction $\gamma$ is too low or $T_c$ is too high.
:::


(sec-world-model-theory-code)=
## 10. Putting It All Together

:::{div} feynman-prose
Let me give you the complete picture. The table below maps every theoretical concept to its
code implementation and the loss that enforces it. If you understand this table, you understand
the covariant world model.
:::

### Theory-to-Code Correspondence

:::{div} feynman-added
| Theory Concept | Reference | Code Location | Loss Signal |
|---------------|-----------|---------------|-------------|
| WFR world model | {prf:ref}`def-wfr-world-model` | `GeometricWorldModel` (full class) | All Phase 2/3 losses |
| Effective potential | {prf:ref}`def-effective-potential` | `CovariantPotentialNet.force_and_potential` | Geodesic loss + energy conservation |
| BAOAB splitting | {prf:ref}`def-baoab-splitting` | `_baoab_step` | Energy conservation loss (Hamiltonian variance) |
| Boltzmann measure preservation | {prf:ref}`prop-baoab-preserves-boltzmann` | O-step in `_baoab_step` | Energy conservation loss |
| Control field | {prf:ref}`def-the-control-field` | `CovariantControlField.forward` | Geodesic loss (indirectly) |
| Entropic force | {prf:ref}`def-the-entropic-force` | `_analytic_U_and_grad` (hyperbolic drive) | Part of composite force |
| Christoffel symbols | {prf:ref}`prop-explicit-christoffel-symbols-for-poincare-disk` | `christoffel_contraction` in A-step | Geodesic accuracy |
| Mass tensor | {prf:ref}`def-mass-tensor` | `ConformalMetric.conformal_factor` | Momentum regularization |
| Cognitive temperature | {prf:ref}`def-cognitive-temperature` | `self.T_c`, `self.c2` in O-step | O-step noise magnitude |
| Jump intensity | {prf:ref}`prop-jump-intensity-from-value-discontinuity` | `_boltzmann_chart_logits` | Chart transition CE |
| Three-channel latent | {prf:ref}`def-three-channel-latent` | Encoder $\to$ world model interface | Orthogonality + enclosure |
| Causal enclosure | {prf:ref}`def-causal-enclosure` | `EnclosureProbe` in Phase 3 | Enclosure GRL loss |
| Closure defect | {prf:ref}`def-closure-defect` | `defect_acc = acc_full - acc_base` | Enclosure probe diagnostics |
| Capacity metric law | {prf:ref}`thm-capacity-constrained-metric-law` | CFL bounds from `min_length` | Force/velocity/cf squashing |
| Holographic flow | {prf:ref}`cor-recovery-of-holographic-flow` | Radial structure of hyperbolic drive $U(z)$ | Analytic gradient $\partial U/\partial z$ |
| Covariant attention | {ref}`sec-covariant-cross-attention-architecture` | All sub-modules use `CovariantAttention` | Geometric consistency |
| Attentive atlas | {ref}`sec-tier-the-attentive-atlas` | `ChartTokenizer.chart_centers` + binding | Phase 2 `bind_chart_centers` |
| Equations of motion | {ref}`sec-the-equations-of-motion-geodesic-jump-diffusion` | `_baoab_step` + `_conditional_jump` | Full Phase 2 loss |
| Jump-diffusion SDE | {ref}`sec-the-coupled-jump-diffusion-sde` | Forward rollout loop | WFR decomposition |
| Stochastic action | {ref}`sec-the-stochastic-action-principle` | Onsager-Machlup via geodesic diffusion | Supervised waypoint loss |
| Unified effective potential | {ref}`sec-the-unified-effective-potential` | $\Phi_{\text{eff}} = \alpha U + (1-\alpha)V + \gamma_{\text{risk}} \Psi$ | Energy conservation |
| Belief dynamics | {ref}`sec-belief-dynamics-prediction-update-projection` | Chart logits + routing update | Chart CE + Zeno loss |
| Discrete filtering | {ref}`sec-filtering-template-on-the-discrete-macro-register` | `DynamicsTransitionModel` | Transition CE |
| Optimizer theory | {ref}`sec-optimizer-thermodynamic-governor` | 3-phase pipeline + alternating optimization | Gradient surgery |
| Architecture | {ref}`sec-the-disentangled-variational-architecture-hierarchical-latent-separation` | Modular sub-module design | All losses |
| Training losses theory | {ref}`sec-loss-function-enforcing-macro-micro-separation` | `compute_phase2_loss`, `compute_phase3_loss` | Weighted loss composition |
:::

### Summary Diagram

```{mermaid}
flowchart TB
    subgraph Phase1["Phase 1: Encoder Training"]
        P1E["Encoder + Decoder"]
        P1L["Recon + VQ + Routing + Ortho + Jump"]
        P1E --> P1L
    end

    subgraph Phase2["Phase 2: World Model Training"]
        P2E["Frozen Encoder"]
        P2W["GeometricWorldModel"]
        P2L["Geodesic + Chart CE + Mom + Energy + Hodge + PDE"]
        P2E --> |"z_all, rw_all, K_all"| P2W
        P2W --> P2L
    end

    subgraph Phase3["Phase 3: Joint Fine-Tuning"]
        P3E["Encoder (0.1x scale)"]
        P3W["World Model (1.0x scale)"]
        P3P["Enclosure Probe"]
        P3E --> |"detach"| P3W
        P3E --> P3P
        P3ALT["Alternating<br/>Optimization"]
        P3E --> P3ALT
        P3W --> P3ALT
        P3P --> P3ALT
    end

    Phase1 --> |"chart centers"| Phase2
    Phase2 --> |"bind_chart_centers"| Phase3
```

### Default Configuration

:::{div} feynman-added
| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 16 | Poincare ball dimension |
| `action_dim` | 6 | Action vector dimension |
| `num_charts` | 8 | Number of atlas charts |
| `codes_per_chart` | 32 | VQ codes per chart |
| `wm_dt` | 0.01 | Integration time step |
| `wm_gamma_friction` | 1.0 | Langevin friction coefficient |
| `wm_T_c` | 0.1 | Cognitive temperature |
| `wm_alpha_potential` | 0.5 | Analytic drive vs. learned critic balance |
| `wm_beta_curl` | 0.1 | Curl field coupling strength |
| `wm_gamma_risk` | 0.01 | Risk force weight |
| `wm_use_boris` | True | Enable Boris rotation |
| `wm_use_jump` | True | Enable conditional chart jumps |
| `wm_refine_steps` | 3 | BAOAB sub-steps per horizon step |
| `wm_min_length` | 0.03 | Minimum resolvable length scale (CFL) |
| `phase1_epochs` | 100 | Encoder training epochs |
| `phase2_epochs` | 50 | World model training epochs |
| `phase3_epochs` | 30 | Joint fine-tuning epochs |
:::

:::{div} feynman-prose
And there it is. A world model that respects the geometry of its latent space, separates smooth
flow from discrete jumps, and enforces six physically motivated constraints through its loss
functions. Every force is covariant. Every integration step is geodesic. Every CFL bound
comes from one length scale. And the three-phase training pipeline builds each layer of
complexity on a stable foundation.

The beautiful thing is not any single piece --- it is how they fit together. The encoder gives
the world model an atlas. The world model learns forces on that atlas. The forces respect the
metric. The metric respects the chart structure. The chart structure respects the codebook.
And in Phase 3, all of these co-adapt under the watchful eye of the enclosure probe, which
makes sure the information goes where the theory says it should go.

That is the covariant geometric world model. It is a Hamiltonian system on a curved space
with discrete jumps, trained in three phases, diagnosed by its own Hodge decomposition. If
you understand how each piece works and why it is there, you understand the dynamics engine
of the entire agent.
:::
