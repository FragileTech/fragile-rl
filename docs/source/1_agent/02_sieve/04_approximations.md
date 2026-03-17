(sec-infeasible-implementation-replacements)=
# Infeasible Implementation Replacements

## TLDR

- Many “ideal” stability/geometry criteria are **computationally intractable**; this chapter provides cheap surrogates
  that preserve the same failure-detection intent.
- Each replacement is a **mapping**: (theoretical barrier/diagnostic) → (practical probe/loss) with an operational
  interpretation and a PyTorch-friendly form.
- Use these surrogates to keep the Sieve runnable in real systems: online checks (✓) plus amortized/offline probes (⚡/✗).
- The goal is *not* to weaken the theory; it is to make the same contracts **implementable** without pretending you can
  compute impossible objects.
- Treat the summary table as an engineering index: it tells you what to compute when a given theoretical requirement is
  out of reach.

## Roadmap

1. Replacement patterns (what gets approximated and why).
2. A catalog of concrete substitutions with code-ready formulas.
3. A summary mapping from theory labels to implementation losses/probes.

(rb-practical-substitutions)=
:::{admonition} Researcher Bridge: Practical Substitutions for Idealized Laws
:class: tip
Many theoretical constraints are too expensive to compute directly. This section provides the RL-engineering replacements (surrogate losses, probes, and bounds) that preserve the same failure detection in practice.
:::

:::{div} feynman-prose
Here is a situation that comes up constantly in physics and engineering: you derive a beautiful, exact criterion for detecting when something goes wrong, then realize that computing it would take longer than the age of the universe. What do you do?

You find a cheaper test that catches the same failures. This is not cheating---it is the essence of good engineering. To know if my car engine is overheating, I do not need the temperature of every molecule. A single thermometer in the coolant tells me what I need.

The theoretical framework gives us exact criteria for instability, bifurcations, and non-tame dynamics. These are mathematically elegant but computationally ruinous. So we ask: what simpler measurement triggers an alarm at the same moments? What is the "thermometer" for each kind of failure?

For each expensive theoretical test, we find a cheap surrogate that rings the same warning bells.
:::

Several regularization terms from the theoretical framework are computationally infeasible for standard training. This section provides practical alternatives with full PyTorch implementations.

(sec-barrierbode-temporal-gain-margin)=
## BarrierBode → Temporal Gain Margin

:::{div} feynman-prose
The Bode sensitivity integral from control theory says something remarkable: you cannot suppress disturbances at all frequencies simultaneously. Push down the response at one frequency, it pops up somewhere else. The integral of log-sensitivity over all frequencies is constant---like conservation of energy, but for control systems.

Why care about this for neural policies? It detects instability. If your controller oscillates wildly, if errors amplify instead of shrink, the Bode integral catches it.

The catch: computing that integral requires the transfer function $S(j\omega)$, which assumes a linear time-invariant system. Neural networks are neither. Even with FFT approximations, we would need long, stationary trajectories---but our agents are constantly exploring and changing.

What are we really trying to detect? Errors getting bigger over time. Oscillations that grow instead of decay. We do not need Fourier analysis for that---we can watch the error magnitudes directly.
:::

**Original (Infeasible):**

$$
\int_{0}^{\infty} \log \lvert S(j\omega) \rvert d\omega = \text{const.} \quad \text{(Bode sensitivity integral)}

$$
**Problem:** Requires frequency-domain analysis of the closed-loop transfer function $S(j\omega)$. Neural policies don't have closed-form transfer functions, and FFT requires long stationary trajectories.

**Replacement: Temporal Gain Margin**

$$
\mathcal{L}_{\text{gain}} = \sum_{k=1}^{K} \max\left(0, \frac{\Vert e_{t+k} \Vert}{\Vert e_t \Vert + \epsilon} - G_{\max}\right)^2

$$

:::{div} feynman-prose
The idea is simple: compare the error at time $t$ to the error at time $t+k$. If the ratio exceeds $G_{\max}$, errors are amplifying---bad news. Summing over horizons $k = 1, 2, \ldots, K$ catches both fast oscillations (small $k$) and slower instabilities (larger $k$).

The squared penalty means small violations get a tap, big violations get hammered. Differentiable everywhere, and it focuses the optimizer on the worst cases.

The default $G_{\max} = 2$ tolerates occasional error doubling (transient disturbances happen), but flags anything worse.
:::

This surrogate loss penalizes error amplification and oscillatory instability without requiring LTI assumptions.

```python
def compute_gain_margin_loss(
    errors: torch.Tensor,  # Shape: [B, T] - tracking errors over time
    G_max: float = 2.0,     # Maximum allowed gain
    K: int = 5,             # Lookahead horizon
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    BarrierBode replacement: Temporal gain margin constraint.

    Penalizes trajectories where errors amplify over time,
    corresponding to the loop instability detected by Bode sensitivity analysis.

    Args:
        errors: [B, T] tensor of error magnitudes at each timestep
        G_max: Maximum allowed amplification ratio
        K: Number of steps to check ahead
        eps: Numerical stability

    Returns:
        Scalar loss penalizing gain violations
    """
    B, T = errors.shape
    if T <= K:
        return torch.tensor(0.0, device=errors.device)

    total_violation = 0.0
    for k in range(1, min(K + 1, T)):
        # Gain at lag k: ||e_{t+k}|| / ||e_t||
        e_t = errors[:, :-k]  # [B, T-k]
        e_t_plus_k = errors[:, k:]  # [B, T-k]

        gain = e_t_plus_k / (e_t + eps)
        violation = torch.relu(gain - G_max).pow(2)
        total_violation = total_violation + violation.mean()

    return total_violation / K


# Alternative: Peak gain detection
def compute_peak_gain_loss(
    errors: torch.Tensor,  # [B, T]
    G_max: float = 2.0,
) -> torch.Tensor:
    """Simplified variant: penalizes maximum gain ratio."""
    B, T = errors.shape
    e_ratios = errors[:, 1:] / (errors[:, :-1] + 1e-6)
    max_gain = e_ratios.max(dim=-1).values  # [B]
    return torch.relu(max_gain - G_max).pow(2).mean()
```

(sec-bifurcatecheck-stochastic-jacobian-probing)=
## BifurcateCheck → Stochastic Jacobian Probing

:::{div} feynman-prose
At a bifurcation, the qualitative behavior of a system changes suddenly---a stable fixed point becomes unstable, splits in two, or starts oscillating. Like water turning to ice: cross a threshold and everything is different.

For a world model $S_t$ predicting latent state evolution, bifurcations spell danger. The model sits on a knife-edge between behaviors, and small input changes send predictions wildly off course.

The signature of bifurcation is in the Jacobian $J_{S_t} = \partial S_t(z) / \partial z$. When an eigenvalue crosses the unit circle (discrete time) or imaginary axis (continuous time), you have a bifurcation.

But the full Jacobian costs $O(Z^2)$ to form and $O(Z^3)$ for eigenvalues. For latent dimension 256, that is millions of operations per sample, every training step, every batch element. Not practical.
:::

**Original (Infeasible):**

$$
\det(J_{S_t}) \quad \text{where } J_{S_t} = \frac{\partial S_t(z)}{\partial z}

$$
**Problem:** Computing the full Jacobian is $O(Z^3)$. For $Z = 256$, this is ~16M operations per sample.

**Replacement: Hutchinson-style Jacobian Probing**

$$
\mathcal{L}_{\text{bifurcate}} = \text{Var}_v\left[\Vert J_{S_t} v \Vert^2\right] \quad \text{where } v \sim \mathcal{N}(0, I)

$$

:::{div} feynman-prose
The trick: we do not need all eigenvalues, just whether they are suspiciously spread out. If eigenvalues are similar, the Jacobian stretches all directions roughly equally. If some are huge and others tiny, different random directions get stretched by wildly different amounts.

Instead of computing the full Jacobian, probe it with random vectors. Pick a random direction $v$, compute $Jv$ (just a gradient computation, cheap with autodiff), measure how much $v$ got stretched. Repeat a few times.

If stretching amounts are similar, eigenvalues are clustered---probably safe. If they vary wildly, eigenvalue spread signals bifurcation sensitivity.

This is the Hutchinson trace estimator from numerical linear algebra: peek at a matrix's spectral properties without ever forming it explicitly.
:::

High variance in the Jacobian-vector product norm indicates instability (eigenvalue spread).

```python
def compute_bifurcation_loss(
    world_model: nn.Module,
    z: torch.Tensor,           # [B, Z] - current latent states
    a: torch.Tensor,           # [B, A] - actions (if needed)
    n_probes: int = 5,
    instability_threshold: float = 1.0,
) -> torch.Tensor:
    """
    BifurcateCheck replacement: Stochastic Jacobian probing.

    Uses Hutchinson trace estimator principle: instead of computing
    full Jacobian, probe with random vectors. High variance in
    ||J @ v|| indicates eigenvalue spread → bifurcation sensitivity.

    Args:
        world_model: S_t(z, a) -> z_next
        z: Current latent states [B, Z]
        a: Actions [B, A]
        n_probes: Number of random direction probes
        instability_threshold: Variance threshold for penalty

    Returns:
        Scalar loss penalizing high Jacobian variance
    """
    B, Z = z.shape
    z = z.requires_grad_(True)

    # Forward through world model
    z_next = world_model(z, a)  # [B, Z]

    vjp_norms = []
    for _ in range(n_probes):
        # Random probe direction
        v = torch.randn_like(z)  # [B, Z]

        # Vector-Jacobian product (VJP) via autodiff (efficient: O(Z))
        vjp = torch.autograd.grad(
            outputs=z_next,
            inputs=z,
            grad_outputs=v,
            create_graph=True,
            retain_graph=True,
        )[0]  # [B, Z]

        vjp_norm = vjp.norm(dim=-1)  # [B]
        vjp_norms.append(vjp_norm)

    # Stack and compute variance across probes
    vjp_norms = torch.stack(vjp_norms, dim=0)  # [n_probes, B]
    variance = vjp_norms.var(dim=0).mean()  # Average variance across batch

    # Penalize high variance (indicates instability)
    loss = torch.relu(variance - instability_threshold).pow(2)

    return loss
```

(sec-tamecheck-lipschitz-gradient-proxy)=
## TameCheck → Lipschitz Gradient Proxy

:::{div} feynman-prose
What does "tame" mean for a function? No sharp corners or sudden kinks. The output changes smoothly through input space---not just in value, but in slope.

The Hessian (second derivatives) captures this smoothness. Bounded Hessian norm means the gradient cannot change too fast---the function is tame. This matters for optimization: gradient descent on non-tame functions oscillates wildly or gets stuck in pathological regions.

But the Hessian is even more expensive than the Jacobian: $O(Z^2 \times P)$ for a world model with $P$ parameters on $Z$-dimensional latent space. Completely impractical.
:::

**Original (Infeasible):**

$$
\Vert \nabla^2 S_t \Vert \quad \text{(Hessian norm)}

$$
**Problem:** Full Hessian is $O(Z^2 \times P_{WM})$ — prohibitive for large world models.

**Replacement: Lipschitz of Gradient**

$$
\mathcal{L}_{\text{tame}} = \frac{\Vert \nabla_z S_t(z_1) - \nabla_z S_t(z_2) \Vert}{\Vert z_1 - z_2 \Vert + \epsilon}

$$

:::{div} feynman-prose
Key insight: bounded Hessian means the gradient does not change too fast as you move through space. That is exactly a Lipschitz condition on the gradient.

So check the Lipschitz constant directly. Take two nearby points $z_1$ and $z_2$. Compute the gradient at each. Measure how much the gradient changed relative to how much the input changed. Bounded ratio means bounded Hessian---that is literally what the Hessian measures.

Computing $\nabla_z S_t(z)$ at one point is cheap (one backward pass). We need two gradients plus a tiny perturbation. The whole operation is $O(Z)$ instead of $O(Z^2)$.

General pattern: when you cannot afford the whole matrix, probe its action on carefully chosen vectors.
:::

Bounded gradient Lipschitz constant implies bounded Hessian (by definition).

```python
def compute_tame_loss(
    world_model: nn.Module,
    z: torch.Tensor,         # [B, Z]
    a: torch.Tensor,         # [B, A]
    perturbation_scale: float = 0.01,
    lipschitz_target: float = 1.0,
) -> torch.Tensor:
    """
    TameCheck replacement: Lipschitz gradient constraint.

    Instead of computing full Hessian, we estimate the Lipschitz
    constant of the gradient via finite differences. This bounds
    the Hessian spectral norm (tameness).

    Args:
        world_model: S_t(z, a) -> z_next
        z: Current latent states [B, Z]
        a: Actions [B, A]
        perturbation_scale: Size of random perturbation
        lipschitz_target: Target Lipschitz constant

    Returns:
        Scalar loss penalizing non-tame dynamics
    """
    B, Z = z.shape

    # Two nearby points
    z1 = z.requires_grad_(True)
    delta = torch.randn_like(z) * perturbation_scale
    z2 = (z + delta).requires_grad_(True)

    # Forward passes
    z1_next = world_model(z1, a)
    z2_next = world_model(z2, a)

    # Compute gradients at both points
    # Sum over output dims to get [B, Z] gradient
    grad1 = torch.autograd.grad(
        z1_next.sum(), z1, create_graph=True, retain_graph=True
    )[0]  # [B, Z]

    grad2 = torch.autograd.grad(
        z2_next.sum(), z2, create_graph=True, retain_graph=True
    )[0]  # [B, Z]

    # Lipschitz estimate: ||grad1 - grad2|| / ||z1 - z2||
    grad_diff = (grad1 - grad2).norm(dim=-1)  # [B]
    z_diff = delta.norm(dim=-1) + 1e-6  # [B]

    lipschitz_estimate = grad_diff / z_diff  # [B]

    # Penalize exceeding target Lipschitz constant
    loss = torch.relu(lipschitz_estimate - lipschitz_target).pow(2).mean()

    return loss
```

(sec-topocheck-value-gradient-alignment)=
## TopoCheck → Value Gradient Alignment

:::{div} feynman-prose
Reachability is fundamental: can I get from here to there? From state $z$ to $z_{\text{goal}}$, does a path exist?

The theoretical answer requires planning: simulate all trajectories, find which reach the goal. For horizon $H$, batch size $B$, and latent dimension $Z$, this costs $O(H \times B \times Z)$---and $H$ might need to be huge to guarantee finding a path.

Why care about reachability? We need the value function to tell the truth. If $V(z)$ says "this state is valuable," there had better be an actual path to high-reward regions. If the latent space has holes or barriers, the value function might look smooth and encouraging while the goal is actually unreachable.
:::

**Original (Infeasible):**

$$
T_{\text{reach}}(z_{\text{goal}}) \quad \text{(Reachability time)}

$$
**Problem:** Requires multi-step planning through world model: $O(H \times B \times Z)$ with potentially large horizon $H$.

**Replacement: Value Gradient Alignment**

$$
\mathcal{L}_{\text{topo}} = \mathrm{ReLU}\left(\left\langle \nabla_z V(z), \frac{z_{\text{goal}} - z}{\| z_{\text{goal}} - z \|} \right\rangle\right)

$$

:::{div} feynman-prose
A much cheaper test: does the value function's gradient point the right way for cost minimization? If $V$ is a cost minimized at the goal, gradient descent follows $-\nabla_z V(z)$. That means $\nabla_z V(z)$ should point away from $(z_{\text{goal}} - z)$ so that $-\nabla_z V(z)$ points toward the goal.

The inner product $\langle \nabla_z V, \hat{d}_{\text{goal}} \rangle$ measures alignment: positive means the gradient points toward the goal (wrong for a cost), negative means it points away (correct). We penalize positive alignment and leave negative alignment alone.

This is necessary but not sufficient for reachability. If you cannot start moving the right direction, you certainly cannot arrive. Obstacles might still block the path---but this catches the common failure where the value function points the wrong way entirely.

Cost: one gradient computation. No multi-step simulation.
:::

When $-\nabla_z V(z)$ aligns with the goal direction, gradient descent on $V$ yields a path to $z_{\text{goal}}$.

```python
def compute_topo_loss(
    critic: nn.Module,
    states: torch.Tensor,      # [B, Z] - current states
    goal_states: torch.Tensor,  # [B, Z] or [Z] - goal states
) -> torch.Tensor:
    """
    TopoCheck replacement: Value gradient alignment.

    Instead of computing multi-step reachability, we check if
    the critic's value gradient points toward the goal. This is
    a necessary condition for gradient-based reachability.

    Args:
        critic: V(z) -> scalar value
        states: Current states [B, Z]
        goal_states: Target states [B, Z] or [Z]

    Returns:
        Scalar loss (negative = gradient points toward goal)
    """
    B, Z = states.shape
    states = states.requires_grad_(True)

    # Compute value and its gradient
    values = critic(states)  # [B]
    grad_v = torch.autograd.grad(
        values.sum(), states, create_graph=True
    )[0]  # [B, Z]

    # Direction to goal
    if goal_states.dim() == 1:
        goal_states = goal_states.unsqueeze(0).expand(B, -1)

    to_goal = goal_states - states  # [B, Z]
    to_goal_normalized = to_goal / (to_goal.norm(dim=-1, keepdim=True) + 1e-6)

    # Alignment: should be negative (V decreases toward goal)
    alignment = (grad_v * to_goal_normalized).sum(dim=-1)  # [B]

    # Loss: penalize positive alignment (wrong direction)
    loss = torch.relu(alignment).mean()

    return loss
```

(sec-geomcheck-efficient-infonce)=
## GeomCheck → Efficient InfoNCE

:::{div} feynman-prose
Contrastive learning: related points should be close in latent space, unrelated points far apart. For temporal data, "related" means close in time---frame $t$ and frame $t+k$ should map to nearby codes.

InfoNCE is the standard loss. The probability of correctly identifying which $z_{t+k}$ goes with $z_t$, among all batch elements, should be high. Mathematically, a softmax over all pairwise similarities.

The problem: "all pairwise." Batch size $B$ means $B^2$ similarity computations. For $B = 1024$ and $Z = 256$, that is a quarter billion multiplications per batch.
:::

**Original (Expensive):**

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_t, z_{t+k}))}{\sum_{j=1}^{B} \exp(\text{sim}(z_t, z_j))}

$$
**Problem:** Full pairwise computation is $O(B^2 \times Z)$.

**Replacement: Sampled InfoNCE**

$$
\mathcal{L}_{\text{InfoNCE}}^{\text{eff}} = -\log \frac{\exp(\text{sim}(z_t, z_{t+k}))}{\exp(\text{sim}(z_t, z_{t+k})) + \sum_{j=1}^{K} \exp(\text{sim}(z_t, z_{\text{neg},j}))}

$$

:::{div} feynman-prose
We do not need every sample as a negative---just enough to make the task challenging. If the encoder distinguishes the true positive from 128 random negatives, it has learned something useful. Whether it could beat 1024 does not matter.

Sample $K$ negatives instead of all $B$. Cost drops from $O(B^2)$ to $O(KB)$. With $K = 128$ and $B = 1024$, that is 8x faster.

Where do negatives come from? Two options: other batch samples (in-batch negatives), or a memory bank of codes from previous batches. The memory bank decouples negative count from batch size---small batches, many negatives.

The projection head is optional but helps. Empirically, contrastive learning works better projecting to a different space before computing similarities. Use the original $z$ downstream; the projected versions are just for the loss.
:::

Use $K \ll B$ sampled negatives instead of full batch.

```python
class EfficientInfoNCE(nn.Module):
    """
    GeomCheck replacement: Efficient contrastive loss.

    Uses K sampled negatives instead of full batch pairwise.
    Reduces O(B²Z) to O(KBZ) where K << B.
    """

    def __init__(
        self,
        latent_dim: int,
        n_negatives: int = 128,
        tau: float = 0.1,  # softmax scale
    ):
        super().__init__()
        self.n_negatives = n_negatives
        self.tau = tau

        # Projection head (optional, improves quality)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self,
        z_anchor: torch.Tensor,   # [B, Z] - z_t
        z_positive: torch.Tensor,  # [B, Z] - z_{t+k} (temporally close)
        z_bank: torch.Tensor = None,  # [M, Z] - memory bank for negatives
    ) -> torch.Tensor:
        """
        Compute efficient InfoNCE loss.

        Args:
            z_anchor: Anchor embeddings [B, Z]
            z_positive: Positive pairs [B, Z]
            z_bank: Optional memory bank for negatives [M, Z]

        Returns:
            Scalar contrastive loss
        """
        B, Z = z_anchor.shape

        # Project
        anchor = F.normalize(self.projector(z_anchor), dim=-1)  # [B, Z]
        positive = F.normalize(self.projector(z_positive), dim=-1)  # [B, Z]

        # Sample negatives
        if z_bank is not None and z_bank.shape[0] >= self.n_negatives:
            # Sample from memory bank
            indices = torch.randperm(z_bank.shape[0])[:self.n_negatives]
            negatives = z_bank[indices]  # [K, Z]
            negatives = F.normalize(self.projector(negatives), dim=-1)
        else:
            # Use other batch elements as negatives (in-batch)
            K = min(self.n_negatives, B - 1)
            # Shuffle and take first K (excluding self)
            perm = torch.randperm(B, device=z_anchor.device)
            negatives = anchor[perm[:K]]  # [K, Z]

        # Positive similarity: [B]
        pos_sim = (anchor * positive).sum(dim=-1) / self.tau

        # Negative similarities: [B, K]
        neg_sim = torch.mm(anchor, negatives.T) / self.tau

        # InfoNCE: log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # = pos - log(exp(pos) + sum(exp(neg)))
        # = pos - logsumexp([pos, neg1, neg2, ...])

        # Combine for logsumexp: [B, K+1]
        all_sim = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)

        # Loss: -pos + logsumexp(all)
        loss = -pos_sim + torch.logsumexp(all_sim, dim=-1)

        return loss.mean()


# Usage example:
def compute_geom_loss(
    vae_encoder: nn.Module,
    x_t: torch.Tensor,      # [B, D] - observation at time t
    x_t_plus_k: torch.Tensor,  # [B, D] - observation at time t+k
    info_nce: EfficientInfoNCE,
) -> torch.Tensor:
    """GeomCheck: Contrastive anchoring for latent space."""
    z_t = vae_encoder(x_t)
    z_t_k = vae_encoder(x_t_plus_k)
    return info_nce(z_t, z_t_k)
```

(sec-summary-replacement-mapping)=
## Summary: Replacement Mapping

:::{div} feynman-prose
Five expensive theoretical tests, five cheap surrogates that detect the same failures. The speedups are not incremental---orders of magnitude.

The key insight: you do not need to compute everything, just enough. Enough random probes for eigenvalue spread. Enough negatives for contrastive learning. Enough gradient samples to bound Lipschitz constants. Full computation gives more information, but not more *useful* information for the purpose at hand.

This is a deep principle. In physics: "effective theories"---no need to simulate quarks to understand how a bridge stands. In computer science: "approximation algorithms"---good-enough answers suffice when they are much cheaper. Here: surrogate losses---cheap tests that catch the same failures as expensive exact criteria.
:::

| Original                  | Replacement        | Speedup  | Preserved Property              |
|---------------------------|--------------------|----------|---------------------------------|
| BarrierBode (FFT)         | Temporal Gain      | ~100×    | Detects oscillatory instability |
| BifurcateCheck ($O(Z^3)$) | Jacobian Probing   | ~$Z^2/K$ | Detects eigenvalue spread       |
| TameCheck ($O(Z^2 P)$)    | Lipschitz Gradient | ~$ZP$    | Bounds Hessian norm             |
| TopoCheck ($O(HBZ)$)      | Value Alignment    | ~$H$     | Ensures goal reachability       |
| GeomCheck ($O(B^2 Z)$)    | Sampled NCE        | ~$B/K$   | Preserves slow features         |

:::{note}
:class: feynman-added
A word of caution: these surrogates are not mathematically equivalent to the originals. They are designed to trigger on the same failure modes, but there may be edge cases where one catches something the other misses. In practice, this is rarely a problem---the surrogates are often more robust because they are less sensitive to numerical issues that plague the exact computations.
:::
