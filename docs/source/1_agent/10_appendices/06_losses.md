(sec-appendix-f-loss-terms-reference)=
# {ref}`Appendix F <sec-appendix-f-loss-terms-reference>`: Loss Terms Reference

## TLDR

- Centralizes **37 loss functions and objectives** defined throughout Volume 1 for quick engineering reference.
- **Curved formulations**: All distance-based losses use the Riemannian metric $G_{ij}$ from the capacity-constrained geometry ({ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`). Each entry includes a "Flat limit" showing the standard ML formula recovered when $G_{ij} \to \delta_{ij}$.
- Organized by domain: TopoEncoder losses, supervised topology, control objectives, reward fields, multi-agent, barriers, belief dynamics, self-supervised, imitation, geometric consistency, and meta-learning.
- Each loss includes: formula, parameters, units, purpose, and source section cross-reference.
- Use this as a single reference when implementing the Fragile Agent training pipeline.

(sec-appendix-f-disentangled-vae-losses)=
## F.1 TopoEncoder Losses

These losses train the Attentive Atlas representation stack ({ref}`Section 3.2 <sec-the-disentangled-variational-architecture-hierarchical-latent-separation>`).

:::{prf:definition} F.1.1 (Reconstruction Loss)
:label: def-f-reconstruction-loss

$$
\mathcal{L}_{\text{recon}} = (x - \hat{x})^i \, G_{ij}^{\text{obs}}(x) \, (x - \hat{x})^j
$$

**Parameters:**
- $x, \hat{x}$ – original and reconstructed observations
- $G_{ij}^{\text{obs}}(x)$ – metric tensor on observation space (learned or fixed)

**Purpose:** Ensures charted latents collectively preserve information for reconstruction, with
metric-weighted distances.

**Units:** $[\mathrm{nat}]$ (when scaled appropriately) or metric-weighted MSE.

**Flat limit:** When $G_{ij}^{\text{obs}} = \delta_{ij}$ (identity), recovers standard MSE:
$\|x - \hat{x}\|^2$.

**Source:** {ref}`Section 3.2 <sec-loss-function-enforcing-macro-micro-separation>`,
Definition {prf:ref}`def-total-disentangled-loss`

:::

:::{prf:definition} F.1.2 (Vector Quantization Loss)
:label: def-f-vq-loss

$$
\mathcal{L}_{\text{vq}} = (z_q - z_e)^i \, G_{ij}(z_e) \, (z_q - z_e)^j + \beta \, (z_e - z_q)^i \, G_{ij}(z_q) \, (z_e - z_q)^j
$$

**Parameters:**
- $z_e$ – encoder output (pre-quantization)
- $z_q$ – quantized code embedding $e_{K}$ (per-chart)
- $G_{ij}(z)$ – metric tensor on latent space
- $\beta$ – commitment weight (default 0.25)

**Purpose:** Stabilizes per-chart codebooks. The first term updates code vectors; the second term
encourages the encoder to commit to nearby codes.

**Flat limit:** When $G_{ij} = \delta_{ij}$, recovers standard VQ:
$\|z_q - z_e\|^2 + \beta\|z_e - z_q\|^2$.

**Source:** {ref}`Section 3.2 <sec-architecture-the-disentangled-vq-vae-rnn>`,
Definition {prf:ref}`def-total-disentangled-loss`

:::

:::{prf:definition} F.1.3 (Routing Entropy Loss)
:label: def-f-closure-loss

$$
\mathcal{L}_{\text{entropy}} = -\frac{1}{B}\sum_{b=1}^{B}\sum_{k=1}^{N_c} w_{bk}\,\log(w_{bk} + \epsilon)
$$

**Parameters:**
- $w_{bk}$ – router weights over charts
- $N_c$ – number of charts

**Purpose:** Penalizes diffuse routing. Lower entropy corresponds to sharper chart assignments.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 3.2 <sec-loss-function-enforcing-macro-micro-separation>`

:::

:::{prf:definition} F.1.4 (Consistency Loss)
:label: def-f-slowness-loss

$$
\mathcal{L}_{\text{consistency}} = \frac{1}{B}\sum_{b=1}^{B} \sum_{k=1}^{N_c} w^{\text{enc}}_{bk}\,\log\left(\frac{w^{\text{enc}}_{bk}+\epsilon}{w^{\text{dec}}_{bk}+\epsilon}\right)
$$

**Parameters:**
- $w^{\text{enc}}$ – encoder router weights
- $w^{\text{dec}}$ – decoder router weights

**Purpose:** Aligns encoder and decoder chart usage.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 3.2 <sec-loss-function-enforcing-macro-micro-separation>`

:::

:::{prf:definition} F.1.5 (Window Loss / Grounding)
:label: def-f-nuisance-kl-loss

$$
\mathcal{L}_{\text{window}} = \max\left(0, \epsilon_{\text{ground}} - I(X;K)\right)^2
$$

**Parameters:**
- $I(X;K) = H(K) - H(K|X)$ – mutual information between input and chart assignment
- $\epsilon_{\text{ground}}$ – grounding threshold

**Purpose:** Enforces the stable learning window by requiring chart assignments to carry
information about inputs.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 3.2 <sec-loss-function-enforcing-macro-micro-separation>`

:::

:::{prf:definition} F.1.6 (Per-Chart Code Entropy Loss)
:label: def-f-texture-kl-loss

$$
\mathcal{L}_{\text{code}} = \frac{1}{N_c}\sum_{k=1}^{N_c} \left(\log K - H(C\mid K=k)\right)
$$

**Parameters:**
- $C$ – code index within each chart
- $K$ – number of codes per chart

**Purpose:** Encourages each chart to use its codebook uniformly rather than collapsing to a
subset.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 3.2 <sec-loss-function-enforcing-macro-micro-separation>`

:::

:::{prf:definition} F.1.7 (Total TopoEncoder Loss)
:label: def-f-total-disentangled-loss

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{vq}} +
\lambda_{\text{ent}}\,\mathcal{L}_{\text{entropy}} +
\lambda_{\text{cons}}\,\mathcal{L}_{\text{consistency}} +
\sum_{i \in \text{tiers}} \lambda_i \mathcal{L}_i +
\lambda_{\text{jump}}\,\mathcal{L}_{\text{jump}} +
\lambda_{\text{sup}}\,\mathcal{L}_{\text{sup}}
$$

**Purpose:** Compound loss enforcing sharp routing, charted quantization, and stable geometry.

**Source:** {ref}`Section 3.2 <sec-loss-function-enforcing-macro-micro-separation>`,
Definition {prf:ref}`def-total-disentangled-loss`

:::

:::{prf:definition} F.1.8 (Jump Consistency Loss)
:label: def-f-overlap-consistency

$$
\mathcal{L}_{\text{jump}} = \mathbb{E}_{i \ne j}\left[\|\,z_n^{(j)} - \mathcal{J}_{i \to j}(z_n^{(i)})\,\|^2\right]
$$

**Parameters:**
- $z_n^{(i)}$ – nuisance coordinate from chart $i$
- $\mathcal{J}_{i \to j}$ – learned jump operator from chart $i$ to chart $j$

**Purpose:** Enforces consistency in chart overlaps by learning transitions between chart-local
nuisance coordinates.

**Units:** Dimensionless (metric-weighted embedding distance).

**Source:** {ref}`Section 7 <sec-the-overlap-consistency-loss>`

:::

## F.2 Supervised Topology Losses

These losses enforce geometric coherence of classification ({ref}`Section 25 <sec-supervised-topology-semantic-potentials-and-metric-segmentation>`).

:::{prf:definition} F.2.1 (Purity Loss / Conditional Entropy)
:label: def-f-purity-loss

$$
\mathcal{L}_{\text{purity}} = \sum_{k=1}^{N_c} P(K=k) \cdot H(Y \mid K=k) = H(Y \mid K)
$$

**Parameters:**
- $P(K=k) = \mathbb{E}_{x \sim \mathcal{D}}[w_k(x)]$ – marginal chart probability
- $H(Y \mid K=k) = -\sum_y P(Y=y \mid K=k) \log P(Y=y \mid K=k)$ – class entropy within chart $k$

**Purpose:** Measures how well charts separate classes. Low purity loss means each chart is associated with a single class. Equivalent to maximizing mutual information $I(K; Y)$ since $\mathcal{L}_{\text{purity}} = H(Y) - I(K; Y)$.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 25.4 <sec-the-supervised-topology-loss>`, Definition {prf:ref}`def-purity-loss`

:::

:::{prf:definition} F.2.2 (Load Balance Loss)
:label: def-f-balance-loss

$$
\mathcal{L}_{\text{balance}} = D_{\text{KL}}\left(\bar{w} \;\|\; \text{Uniform}(N_c)\right)
$$

**Parameters:**
- $\bar{w} = \mathbb{E}_{x \sim \mathcal{D}}[w(x)]$ – average router weight vector
- $N_c$ – number of charts

**Purpose:** Prevents "dead charts" (collapse to few charts). Encourages all charts to be used productively, addressing the expert-collapse problem in mixture-of-experts systems.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 25.4 <sec-load-balance-loss>`, Definition {prf:ref}`def-balance-loss`

:::

:::{prf:definition} F.2.3 (Metric Contrastive Loss)
:label: def-f-contrastive-loss

$$
\mathcal{L}_{\text{metric}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}: y_i \neq y_j} w_i^\top w_j \cdot \max(0, m - d_G^{\text{jump}}(z_i, z_j))^2
$$

**Parameters:**
- $\mathcal{P}$ – set of sample pairs in batch
- $w_i, w_j$ – router weight vectors
- $m > 0$ – margin (minimum desired geodesic separation)
- $d_G^{\text{jump}}(z_i, z_j)$ – minimum geodesic jump cost between samples under metric $G$

**Purpose:** Enforces that different-class samples are geometrically far apart in geodesic jump distance. The weighting $w_i^\top w_j$ focuses penalty on hard examples (high routing overlap despite different classes). The geodesic distance respects the curved manifold structure.

**Units:** $[\mathrm{nat}]$

**Flat limit:** When $G_{ij} = \delta_{ij}$, reduces to Euclidean jump distance.

**Source:** {ref}`Section 25.4 <sec-metric-contrastive-loss>`, Definition {prf:ref}`def-contrastive-loss`

:::

:::{prf:definition} F.2.4 (Route Alignment Loss)
:label: def-f-route-alignment-loss

$$
\mathcal{L}_{\text{route}} = \mathbb{E}_{x, y_{\text{true}}}\left[\text{CE}\left(\sum_k w_k(x) \cdot P(Y=\cdot \mid K=k), \; y_{\text{true}}\right)\right]
$$

**Parameters:**
- $w_k(x)$ – router weights for sample $x$ and chart $k$
- $P(Y=\cdot \mid K=k)$ – per-chart class distributions
- $\text{CE}$ – cross-entropy loss

**Purpose:** Primary classification loss. The predicted class distribution (router-weighted average of per-chart distributions) must match the true label.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 25.4 <sec-route-alignment-loss>`, Definition {prf:ref}`def-route-alignment-loss`

:::

:::{prf:definition} F.2.5 (Combined Supervised Topology Loss)
:label: def-f-total-supervised-loss

$$
\mathcal{L}_{\text{sup-topo}} = \mathcal{L}_{\text{route}} + \lambda_{\text{pur}} \mathcal{L}_{\text{purity}} + \lambda_{\text{bal}} \mathcal{L}_{\text{balance}} + \lambda_{\text{met}} \mathcal{L}_{\text{metric}}
$$

**Typical hyperparameters:**
| Weight | Typical Value | Role |
|--------|---------------|------|
| $\lambda_{\text{pur}}$ | 0.1 | Chart purity |
| $\lambda_{\text{bal}}$ | 0.01 | Load balancing |
| $\lambda_{\text{met}}$ | 0.01 | Metric separation |

**Purpose:** Weighted combination enforcing chart purity, balanced usage, geometric separation, and prediction accuracy.

**Source:** {ref}`Section 25.4 <sec-combined-supervised-topology-loss>`, Definition {prf:ref}`def-total-loss`

:::

:::{prf:definition} F.2.6 (Hierarchical Supervised Loss)
:label: def-f-hierarchical-loss

$$
\mathcal{L}_{\text{hier}} = \sum_{\ell=0}^{L} \alpha_\ell \left(\mathcal{L}_{\text{route}}^{(\ell)} + \lambda_{\text{pur}} \mathcal{L}_{\text{purity}}^{(\ell)}\right)
$$

**Parameters:**
- $\ell \in \{0, \ldots, L\}$ – scale levels (bulk to boundary)
- $\alpha_\ell$ – per-scale weights (often $\alpha_\ell = 1$ or decaying)
- $\mathcal{Y}_\ell$ – label space at scale $\ell$ (coarse to fine)

**Purpose:** Enforces classification at multiple scales via stacked TopoEncoders. Coarse (bulk) layers distinguish broad categories; fine (boundary) layers distinguish leaf categories.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 25.6 <sec-hierarchical-classification-via-scale-decomposition>`, Definition {prf:ref}`def-hierarchical-supervised-loss`

:::

(sec-appendix-f-control-and-value-objectives)=
## F.3 Control and Value Objectives

These objectives define control, value, and reward structure ({ref}`Section 2 <sec-the-control-loop-representation-and-control>`, {ref}`Section 18 <sec-the-reward-field-value-forms-and-hodge-geometry>`).

:::{prf:definition} F.3.1 (Cumulative Cost Functional)
:label: def-f-cumulative-cost

$$
\mathcal{S} = \int \Big(\mathcal{L}_{\text{control}} + C(z_t, a_t)\Big) \, dt
$$

**Parameters:**
- $\mathcal{L}_{\text{control}}$ – control/effort cost (KL penalty, action magnitude)
- $C(z_t, a_t)$ – task cost

**Purpose:** General optimal control objective under information/effort constraints. Specializes to KL-control and entropy-regularized RL.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 2 <sec-the-control-loop-representation-and-control>`

:::

:::{prf:definition} F.3.2 (Instantaneous Regularized Objective)
:label: def-f-instantaneous-objective

$$
F_t := V(Z_t) + \beta_K\big(-\log p_\psi(K_t)\big) + \beta_n D_{\mathrm{KL}}(q(z_{n,t} \mid x_t) \| p(z_n)) + \beta_{\mathrm{tex}} D_{\mathrm{KL}}(q(z_{\mathrm{tex},t} \mid x_t) \| p(z_{\mathrm{tex}})) + T_c D_{\mathrm{KL}}(\pi(\cdot \mid K_t) \| \pi_0(\cdot \mid K_t))
$$

**Parameters:**
- $V(Z_t)$ – task-aligned cost-to-go (critic estimate)
- $\beta_K(-\log p_\psi(K_t))$ – macro codelength penalty (Occam's razor for discrete state)
- $\beta_n, \beta_{\text{tex}}$ – residual regularization weights
- $T_c$ – cognitive temperature
- $\pi_0$ – prior policy

**Purpose:** Trades off task cost, representation complexity, and control effort, all in consistent units (nats).

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 3.2 <sec-the-entropy-regularized-objective-functional>`, Definition {prf:ref}`def-instantaneous-objective`

:::

:::{prf:definition} F.3.3 (Monotonicity Surrogate Loss)
:label: def-f-monotonicity-loss

$$
\mathcal{L}_{\downarrow F} := \mathbb{E}\left[\mathrm{ReLU}(F_{t+1} - F_t)^2\right]
$$

**Purpose:** Penalizes increases in the instantaneous objective $F_t$ from one step to the next, encouraging trajectories that smoothly descend the objective landscape.

**Units:** $[\mathrm{nat}^2]$

**Source:** {ref}`Section 3.2 <sec-the-entropy-regularized-objective-functional>`

:::

:::{prf:definition} F.3.4 (Closure Ratio Diagnostic)
:label: def-f-closure-ratio

$$
\text{Closure Ratio} = \frac{\mathbb{E}[-\log p_\psi(K_{t+1} \mid K_t, a_t)]}{\mathbb{E}[-\log p_{\text{base}}(K_{t+1})]} = \frac{H(K_{t+1} \mid K_t, a_t)}{H(K_{t+1})}
$$

**Interpretation:**
| Ratio | Meaning | Action |
|-------|---------|--------|
| $\ll 1$ | Strong predictive law learned | Success |
| $\approx 1$ | No predictive law | Increase model capacity |
| $> 1$ | Worse than baseline | Bug/degeneracy |

**Purpose:** Measures how much better the macro dynamics model predicts $K_{t+1}$ compared to a marginal baseline. The gap estimates predictive information $I(K_{t+1}; K_t, a_t)$.

**Units:** Dimensionless.

**Source:** {ref}`Section 3.2 <sec-runtime-diagnostics-the-closure-ratio>`, Definition {prf:ref}`def-closure-ratio`

:::

:::{prf:definition} F.3.5 (Causal Information Potential)
:label: def-f-causal-info-potential

$$
\Psi_{\text{causal}}(z, a) := \mathbb{E}_{z' \sim \bar{P}(\cdot | z, a)} \left[ D_{\text{KL}} \left( p(\theta_W | z, a, z') \| p(\theta_W | z, a) \right) \right]
$$

**Parameters:**
- $z, a$ – current state and action
- $z'$ – next state sampled from world model
- $\theta_W$ – world model parameters
- $\bar{P}$ – world model transition distribution

**Purpose:** Measures the **Expected Information Gain** about world model parameters from executing action $a$ at state $z$. High $\Psi_{\text{causal}}$ indicates the outcome will resolve significant uncertainty about dynamics. Drives intrinsic motivation for exploration via Bayesian experimental design.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 29 <sec-the-causal-information-potential>`, Definition {prf:ref}`def-causal-information-potential`

:::

(sec-appendix-f-reward-field-objectives)=
## F.4 Reward Field Objectives

These define reward as a geometric object ({ref}`Section 18 <sec-the-reward-field-value-forms-and-hodge-geometry>`).

:::{prf:definition} F.4.1 (Hodge Decomposition of Reward)
:label: def-f-hodge-decomposition

$$
\mathcal{R} = \underbrace{d\Phi}_{\text{Gradient}} + \underbrace{\delta \Psi}_{\text{Solenoidal}} + \underbrace{\eta}_{\text{Harmonic}}
$$

**Components:**
- $d\Phi$ – Gradient/conservative component (optimizable via value function $V = \Phi$)
- $\delta\Psi$ – Solenoidal/rotational component (cyclic reward structure)
- $\eta$ – Harmonic component (topological cycles from manifold holes)

**Purpose:** Decomposes reward 1-form into orthogonal components. Separates optimizable value from inherently cyclic structure (e.g., Rock-Paper-Scissors).

**Units:** $[\Phi] = \mathrm{nat}$, $[\Psi] = \mathrm{nat} \cdot [\text{length}]^2$, $[\eta] = \mathrm{nat}/[\text{length}]$.

**Source:** {ref}`Section 18.2 <sec-hodge-decomposition-of-value>`, Theorem {prf:ref}`thm-hodge-decomposition`

:::

:::{prf:definition} F.4.2 (Value Curl / Vorticity)
:label: def-f-value-curl

$$
\mathcal{F}_{ij} := \partial_i \mathcal{R}_j - \partial_j \mathcal{R}_i = d\mathcal{R}
$$

**Properties:**
- Antisymmetric: $\mathcal{F}_{ij} = -\mathcal{F}_{ji}$
- Satisfies Bianchi identity: $d\mathcal{F} = 0$
- Gauge-invariant under $\mathcal{R} \to \mathcal{R} + d\chi$

**Purpose:** Detects non-conservative reward structure. Non-zero curl indicates orbiting strategies may be optimal. Diagnostic: $\oint_\gamma \mathcal{R} = \int_\Sigma \mathcal{F} \, d\Sigma \neq 0$ implies non-conservative rewards.

**Units:** $[\mathcal{F}] = \mathrm{nat}/[\text{length}]^2$

**Source:** {ref}`Section 18.2 <sec-hodge-decomposition-of-value>`, Definition {prf:ref}`def-value-curl`

:::

:::{prf:definition} F.4.3 (Class-Conditioned Potential)
:label: def-f-class-potential

$$
V_y(z, K) := -\beta_{\text{class}} \log P(Y=y \mid K) + V_{\text{base}}(z, K)
$$

**Parameters:**
- $P(Y=y \mid K) = \text{softmax}(\Theta_{K,:})_y$ – learnable chart-to-class affinities
- $V_{\text{base}}(z, K)$ – unconditioned critic
- $\beta_{\text{class}} > 0$ – class temperature (inverse semantic diffusion)

**Purpose:** Shapes potential landscape so class-$y$ regions become energy minima. Used for both classification (relaxation inference) and generation (Langevin sampling).

**Units:** $[V_y] = \mathrm{nat}$

**Source:** {ref}`Section 25.2 <sec-the-semantic-potential>`, Definition {prf:ref}`def-class-conditioned-potential`

:::

(sec-appendix-f-multi-agent-losses)=
## F.5 Multi-Agent and Gauge Losses

These govern multi-agent alignment ({ref}`Section 37 <sec-the-inter-subjective-metric-gauge-locking-and-the-emergence-of-objective-reality>`).

:::{prf:definition} F.5.1 (Synchronization Potential)
:label: def-f-sync-potential

$$
\mathcal{L}_{\text{sync}} = \beta \Psi_{\text{sync}} = \beta \int_{\partial\Omega} \mathcal{F}_{AB}^{\mu\nu} \mathcal{F}_{AB\,\mu\nu} \, dA
$$

**Parameters:**
- $\beta$ – coupling strength
- $\mathcal{F}_{AB}$ – Locking curvature (geometric disagreement between agents $A$ and $B$)

**Purpose:** Penalizes disagreement in representations between agents. Drives gauge locking (synchronized metrics). In the strong coupling limit ($\beta \to \infty$), forces $\mathcal{F}_{AB} \to 0$ (perfect synchronization).

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 37 <sec-the-inter-subjective-metric-gauge-locking-and-the-emergence-of-objective-reality>`

:::

(sec-appendix-f-metabolic-losses)=
## F.6 Metabolic and Information Losses

These relate to computation cost and information bounds ({ref}`Section 36 <sec-the-metabolic-transducer-autopoiesis-and-the-szilard-engine>`, {ref}`Section 33 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>`).

:::{prf:definition} F.6.1 (Ontological Stress)
:label: def-f-ontological-stress

$$
\Xi = \sum_{\ell=1}^{L} \left( z_{\text{tex}}^{(\ell)} \right)^i G_{ij}^{(\ell)} \left( z_{\text{tex}}^{(\ell)} \right)^j
$$

**Parameters:**
- $z_{\text{tex}}^{(\ell)}$ – texture embedding at scale $\ell$
- $G_{ij}^{(\ell)}$ – metric tensor at scale $\ell$

**Purpose:** Measures predictability *within* texture across scales. High stress indicates ontological inadequacy---texture contains compressible structure that should have been captured by macro/nuisance. Dual to closure defect: closure measures micro-to-macro leakage; ontological stress measures within-texture predictability.

**Units:** Dimensionless (metric-weighted embedding norm).

**Flat limit:** When $G_{ij}^{(\ell)} = \delta_{ij}$, recovers $\sum_\ell \|z_{\text{tex}}^{(\ell)}\|^2$.

**Source:** {ref}`Section 33 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>`

:::

(sec-appendix-f-barrier-losses)=
## F.7 Barrier and Regularization Losses

These losses enforce fundamental limits and trade-offs ({ref}`Section 4 <sec-4-limits-barriers-the-limits-of-control>`).

:::{prf:definition} F.7.1 (Gradient Penalty Loss)
:label: def-f-gradient-penalty

$$
\mathcal{L}_{GP} = \mathbb{E}_{\hat{s}} \left[\left(\|\nabla_A V\|_G - K\right)^2\right], \qquad \|\nabla_A V\|_G^2 := G^{ij}(\hat{s}) \, (\partial_i V - A_i) \, (\partial_j V - A_j)
$$

**Parameters:**
- $\hat{s}$ – interpolated samples between real and generated
- $V(\hat{s})$ – critic value at sample
- $G^{ij}(\hat{s})$ – inverse metric tensor (contravariant) at sample
- $K$ – target gradient norm (typically 1)

**Purpose:** Enforces Lipschitz constraint on the critic using the metric-induced norm. The covariant gradient norm $\|\nabla_A V\|_G$ measures the gauge-invariant rate of change along geodesics. Prevents vanishing gradients in flat value regions (BarrierGap) and ensures smooth value landscape for stable learning.

**Units:** Dimensionless.

**Flat limit:** When $G^{ij} = \delta^{ij}$ and $A=0$, recovers $(\|\nabla_A V\|_2 - K)^2$.

**Source:** {ref}`Section 4 <sec-barrier-implementation-details>`

:::

:::{prf:definition} F.7.2 (Information-Control Loss)
:label: def-f-info-control

$$
\mathcal{L}_{\text{InfoControl}} = \underbrace{\beta_K \mathbb{E}[-\log p_\psi(K)] + \beta_n D_{\mathrm{KL}}(q(z_n \mid x) \| p(z_n)) + \beta_{\mathrm{tex}} D_{\mathrm{KL}}(q(z_{\mathrm{tex}} \mid x) \| p(z_{\mathrm{tex}}))}_{\text{Compression (Rate)}} + \underbrace{\gamma \mathbb{E}[\mathfrak{D}(Z, A)]}_{\text{Control Effort}}
$$

**Parameters:**
- $\beta_K, \beta_n, \beta_{\text{tex}}$ – compression weights for macro/nuisance/texture
- $\mathfrak{D}(Z, A)$ – actuation cost (KL-control or action norm)
- $\gamma$ – control effort weight

**Purpose:** Balances the Information-Control Tradeoff (BarrierScat vs BarrierCap). High compression removes details needed for fine control; this loss finds the Pareto frontier.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 4 <sec-b-cross-barrier-regularization>`

:::

:::{prf:definition} F.7.3 (Elastic Weight Consolidation)
:label: def-f-ewc

$$
\mathcal{L}_{\text{EWC}} = \sum_i F_i (\theta_i - \theta^*_{i,\text{old}})^2
$$

**Parameters:**
- $F_i$ – diagonal Fisher Information for parameter $i$
- $\theta_i$ – current parameter value
- $\theta^*_{i,\text{old}}$ – parameter value from previous task

**Purpose:** Addresses the Stability-Plasticity Dilemma (BarrierVac vs BarrierPZ). High-sensitivity weights (large $F_i$) are constrained to preserve past learning; low-sensitivity weights can adapt freely.

**Units:** Dimensionless (parameter space distance, Fisher-weighted).

**Source:** {ref}`Section 4 <sec-b-cross-barrier-regularization>`

:::

:::{prf:definition} F.7.4 (Bode Magnitude Loss)
:label: def-f-bode

$$
\mathcal{L}_{\text{Bode}} = \|\mathcal{F}(e_t) \cdot W(\omega)\|^2
$$

**Parameters:**
- $\mathcal{F}(e_t)$ – Fourier transform of error signal
- $W(\omega)$ – frequency weighting function

**Purpose:** Addresses the Bode Sensitivity Integral (BarrierBode). Suppressing error in one frequency band amplifies it in another (waterbed effect). This loss explicitly chooses where to be sensitive vs. blind.

**Units:** $[\mathrm{nat}^2]$

**Source:** {ref}`Section 4 <sec-b-cross-barrier-regularization>`

:::

(sec-appendix-f-belief-dynamics-losses)=
## F.8 Belief Dynamics Losses

These losses enforce belief update constraints ({ref}`Section 11 <sec-belief-dynamics-prediction-update-projection>`).

:::{prf:definition} F.8.1 (Quantum Speed Limit Loss)
:label: def-f-qsl

$$
\mathcal{L}_{\text{QSL}} := \mathrm{ReLU}\left(d_G(z_{t+1}, z_t) - v_{\max}\right)^2
$$

**Parameters:**
- $d_G(z_{t+1}, z_t)$ – geodesic distance traveled in one step
- $v_{\max}$ – maximum allowed velocity in latent space

**Purpose:** Enforces the Quantum Speed Limit: belief cannot change faster than the Mandelstam-Tamm bound allows. Prevents unrealistic jumps in belief state.

**Units:** $[\mathrm{nat}^2]$

**Source:** {ref}`Section 11 <sec-belief-dynamics-prediction-update-projection>`

:::

:::{prf:definition} F.8.2 (Joint Prediction Loss)
:label: def-f-joint-prediction

$$
\mathcal{L}_{\text{joint}} = d_G(\hat{x}_{t+1}^A, x_{t+1})^2 + d_G(\hat{x}_{t+1}^B, x_{t+1})^2 + \beta \Psi_{\text{sync}}
$$

Expanded in coordinates:

$$
d_G(\hat{x}, x)^2 = (\hat{x} - x)^i \, G_{ij}^{\text{obs}}(x) \, (\hat{x} - x)^j
$$

**Parameters:**
- $\hat{x}_{t+1}^A, \hat{x}_{t+1}^B$ – predictions from agents $A$ and $B$
- $x_{t+1}$ – actual next observation
- $G_{ij}^{\text{obs}}$ – metric tensor on observation space
- $\Psi_{\text{sync}}$ – synchronization potential
- $\beta$ – coupling strength

**Purpose:** Multi-agent world model training. Both agents must predict accurately (measured under the observation-space metric), and their representations must synchronize (gauge lock).

**Units:** Metric-weighted prediction error + $[\mathrm{nat}]$.

**Flat limit:** When $G_{ij}^{\text{obs}} = \delta_{ij}$, recovers $\|\hat{x}^A - x\|^2 + \|\hat{x}^B - x\|^2 + \beta\Psi_{\text{sync}}$.

**Source:** {ref}`Section 37 <sec-the-inter-subjective-metric-gauge-locking-and-the-emergence-of-objective-reality>`

:::

(sec-appendix-f-self-supervised-losses)=
## F.9 Self-Supervised and Contrastive Losses

These losses prevent representation collapse without requiring labels ({ref}`Section 3 <sec-diagnostics-stability-checks>`).

:::{prf:definition} F.9.1 (VICReg Loss)
:label: def-f-vicreg

$$
\mathcal{L}_{\text{VICReg}} = \lambda \mathcal{L}_{\text{inv}} + \mu \mathcal{L}_{\text{var}} + \nu \mathcal{L}_{\text{cov}}
$$

**Components:**

$$
\begin{aligned}
\mathcal{L}_{\text{inv}} &= d_G(z, z')^2 = (z - z')^i \, G_{ij}(z) \, (z - z')^j & \text{(invariance)} \\
\mathcal{L}_{\text{var}} &= \frac{1}{d} \sum_{j=1}^{d} \max\left(0, \gamma - \sqrt{\text{Var}_G(z^j) + \epsilon}\right) & \text{(variance)} \\
\mathcal{L}_{\text{cov}} &= \frac{1}{d} \sum_{i \neq j} \left[G^{-1/2} \text{Cov}(z) \, G^{-1/2}\right]_{ij}^2 & \text{(covariance)}
\end{aligned}
$$

**Parameters:**
- $z, z'$ – embeddings of two augmented views of the same input
- $G_{ij}(z)$ – metric tensor on embedding space
- $\gamma$ – variance threshold (typically 1)
- $\lambda, \mu, \nu$ – component weights (typically $\lambda = 25$, $\mu = \nu = 1$)

**Purpose:** Prevents representation collapse without negative samples. Invariance pulls augmented views together (geodesic distance); variance prevents dimension collapse (metric-aware); covariance decorrelates dimensions (whitened by metric).

**Units:** Dimensionless.

**Flat limit:** When $G_{ij} = \delta_{ij}$, recovers standard VICReg with $\|z - z'\|^2$.

**Source:** {ref}`Section 3 <sec-diagnostics-stability-checks>` (GeomCheck, Node 6)

:::

:::{prf:definition} F.9.2 (InfoNCE Loss)
:label: def-f-infonce

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp\left(-d_G(z_t, z_{t+k})^2 / \tau\right)}{\sum_{j} \exp\left(-d_G(z_t, z_j)^2 / \tau\right)}
$$

**Parameters:**
- $z_t, z_{t+k}$ – embeddings at current and future timesteps
- $z_j$ – negative samples (other timesteps or other sequences)
- $d_G(z, z')$ – geodesic distance under metric $G$
- $\tau$ – temperature parameter (squared distance scale)

**Purpose:** Contrastive predictive coding with geodesic similarity. Anchors macro latents to temporal structure by maximizing mutual information between present and future representations. The geodesic kernel $k_G(z, z') = \exp(-d_G(z,z')^2/\tau)$ respects the curved geometry of the latent manifold.

**Units:** $[\mathrm{nat}]$

**Flat limit:** When $G_{ij} = \delta_{ij}$ and using $\text{sim}(z,z') = -\|z-z'\|^2$, recovers standard InfoNCE with Gaussian kernel.

**Source:** {ref}`Section 3 <sec-diagnostics-stability-checks>` (GeomCheck, Node 6)

:::

(sec-appendix-f-imitation-losses)=
## F.10 Imitation and Distillation Losses

These losses train policies from demonstrations or teacher models ({ref}`Section 25 <sec-supervised-topology-semantic-potentials-and-metric-segmentation>`).

:::{prf:definition} F.10.1 (Behavior Cloning Loss)
:label: def-f-bc

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}_{(s, a^*) \sim \mathcal{D}_{\text{expert}}}[-\log \pi(a^* \mid s)]
$$

**Parameters:**
- $(s, a^*)$ – state-action pairs from expert demonstrations
- $\mathcal{D}_{\text{expert}}$ – expert demonstration dataset
- $\pi(a \mid s)$ – learned policy

**Purpose:** Supervised policy learning. Trains the policy to match expert actions via maximum likelihood.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 25 <sec-supervised-topology-semantic-potentials-and-metric-segmentation>`

:::

(sec-appendix-f-geometric-consistency-losses)=
## F.11 Geometric Consistency Losses

These losses enforce geometric laws derived from capacity constraints ({ref}`Section 18 <sec-the-reward-field-value-forms-and-hodge-geometry>`, {ref}`Section 20 <sec-wfr-dynamics-with-memory-sources>`).

:::{prf:definition} F.11.1 (WFR Consistency Loss)
:label: def-f-wfr-consistency

$$
\mathcal{L}_{\text{WFR}} = \left\| \sqrt{\rho_{t+1}} - \sqrt{\rho_t} - \frac{\Delta t}{2\sqrt{\rho_t}}\left(\rho_t r_t - \nabla \cdot (\rho_t v_t)\right) \right\|_{L^2}^2
$$

**Parameters:**
- $\rho_t$ – belief density at time $t$
- $r_t$ – reaction rate (birth/death)
- $v_t$ – transport velocity field
- $\Delta t$ – timestep

**Purpose:** Enforces Wasserstein-Fisher-Rao consistency. Penalizes deviations from the unbalanced continuity equation in cone-space formulation.

**Units:** $[\mathrm{nat}^2]$

**Source:** {ref}`Section 20 <sec-wfr-dynamics-with-memory-sources>`

:::

:::{prf:definition} F.11.2 (Critic TD Loss with PDE Regularization)
:label: def-f-critic-td

$$
\mathcal{L}_{\text{critic}} = \|\text{TD-Error}\|^2 + \lambda_{\text{PDE}} \| -\Delta_G V + \kappa^2 V - \rho_r \|^2
$$

**Parameters:**
- TD-Error $= r + \gamma V(s') - V(s)$ – temporal difference error
- $\Delta_G$ – Laplace-Beltrami operator on manifold
- $\kappa^2 = -\ln \gamma$ – screening mass from discount factor
- $\rho_r$ – reward density

**Purpose:** Combines TD learning with Helmholtz PDE regularization. The PDE term enforces that the critic satisfies the continuum Bellman equation.

**Units:** $[\mathrm{nat}^2]$

**Source:** {ref}`Section 18 <sec-the-reward-field-value-forms-and-hodge-geometry>`

:::

(sec-appendix-f-consensus-losses)=
## F.12 Consensus Losses (Proof of Useful Work)

These support the PoUW consensus mechanism ({ref}`Section 38 <sec-proof-of-useful-work-cognitive-metabolism-as-consensus>`).

:::{prf:definition} F.12.1 (Waste Quotient)
:label: def-f-waste-quotient

$$
W_\mathcal{P} := 1 - \frac{\Delta I_{\text{world}}}{\int \dot{\mathcal{M}}(t) \, dt}
$$

**Parameters:**
- $\Delta I_{\text{world}}$ – mutual information gained about world
- $\dot{\mathcal{M}}(t)$ – metabolic flux (energy dissipation rate)

**Interpretation:**
| Protocol | Waste Quotient | Meaning |
|----------|----------------|---------|
| Bitcoin PoW | $W_{\text{BTC}} \approx 1$ | Energy produces zero world knowledge |
| Target PoUW | $W_{\text{PoUW}} \to 0$ | Energy produces useful learning |

**Purpose:** Measures efficiency of consensus protocol. Low waste quotient means energy dissipation produces useful information gain.

**Units:** Dimensionless.

**Source:** {ref}`Section 38.1 <sec-the-thermodynamic-inefficiency-of-nakamoto-consensus>`, Definition {prf:ref}`def-waste-quotient`

:::

(sec-appendix-f-meta-learning-losses)=
## F.13 Meta-Learning Losses

These losses train the Governor (hyperparameter controller) via bilevel optimization ({ref}`Section 26 <sec-the-universal-governor>`).

:::{prf:definition} F.13.1 (Governor Training Regret)
:label: def-f-governor-regret

$$
J(\phi) = \mathbb{E}_{\mathcal{T} \sim P(\mathcal{T})} \left[ \sum_{t=0}^T \left( \mathcal{L}_{\text{task}}(\theta_t) + \gamma_{\text{viol}} \sum_{k=1}^K \text{ReLU}(C_k(\theta_t))^2 \right) \right]
$$

**Parameters:**
- $\phi$ – Governor parameters
- $\mathcal{T}$ – task from task distribution
- $\mathcal{L}_{\text{task}}(\theta_t)$ – task loss at training step $t$
- $C_k(\theta_t)$ – constraint $k$ value (negative when satisfied)
- $\gamma_{\text{viol}}$ – constraint violation penalty weight

**Purpose:** Meta-learning objective for the Governor. Minimizes cumulative task loss (convergence speed) plus squared constraint violations (feasibility). The Governor learns to set hyperparameters $\Lambda$ that lead to fast, stable training across diverse tasks.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 26 <sec-bilevel-optimization-objective>`, Definition {prf:ref}`def-outer-problem-governor-optimization`

:::

(sec-appendix-f-summary-table)=
## F.14 Summary Table

All distance-based losses use the metric tensor $G_{ij}$. Flat limits recover standard ML formulas when $G_{ij} \to \delta_{ij}$.

| Loss | Sec | Formula Key (Curved) | Purpose |
|------|-----|----------------------|---------|
| **VICReg** | 3 | $d_G(z,z')^2 + \text{var}_G + \text{cov}_G$ | Collapse prevention |
| **InfoNCE** | 3 | $\exp(-d_G^2/\tau)$ kernel | Contrastive prediction |
| **Reconstruction** | 3.2 | $(x-\hat{x})^i G_{ij}^{\text{obs}} (x-\hat{x})^j$ | Information preservation |
| **VQ** | 3.2 | $(z_q-z_e)^i G_{ij} (z_q-z_e)^j$ | Discrete symbol stability |
| **Closure** | 3.2 | $-\log p(K_{t+1} \mid K_t, a)$ | Causal enclosure |
| **Slowness** | 3.2 | $d_G(e_{K_t}, e_{K_{t-1}})^2$ | Anti-symbol-churn |
| **Nuisance KL** | 3.2 | $D_{\text{KL}}(q(z_n) \| \mathcal{N})$ | Structured residual prior |
| **Texture KL** | 3.2 | $D_{\text{KL}}(q(z_{\text{tex}}) \| \mathcal{N})$ | Reconstruction residual |
| **Monotonicity** | 3.2 | $\text{ReLU}(F_{t+1} - F_t)^2$ | Objective descent |
| **Gradient Penalty** | 4 | $(\|\nabla_A V\|_G - K)^2$ | Lipschitz constraint |
| **InfoControl** | 4 | Compression + Control effort | Information-control tradeoff |
| **EWC** | 4 | $\sum_i F_i (\theta_i - \theta^*_i)^2$ | Stability-plasticity balance |
| **Bode** | 4 | $\|\mathcal{F}(e_t) W(\omega)\|^2$ | Frequency sensitivity |
| **Overlap Consistency** | 7 | $(z_n^{(j)} - L_{i→j})^k G^{(j)}_{k\ell} (\cdot)^\ell$ | Chart transition coherence |
| **QSL** | 11 | $\text{ReLU}(d_G - v_{\max})^2$ | Quantum speed limit |
| **Hodge Decomp** | 18 | $\mathcal{R} = d\Phi + \delta\Psi + \eta$ | Reward structure |
| **Value Curl** | 18 | $\mathcal{F} = d\mathcal{R}$ | Non-conservative detection |
| **Critic TD+PDE** | 18 | $\|\text{TD}\|^2 + \lambda\|\Delta_G V - \cdots\|^2$ | Value function learning |
| **WFR Consistency** | 20 | Cone-space continuity | Transport-reaction balance |
| **Purity** | 25 | $H(Y \mid K)$ | Chart semantic purity |
| **Balance** | 25 | $D_{\text{KL}}(\bar{w} \| U)$ | Prevent chart collapse |
| **Metric Contrastive** | 25 | $\max(0, m - d_G^{\text{jump}})^2$ | Geometric separation |
| **Route Alignment** | 25 | $\text{CE}(\sum_k w_k P(Y \mid K), y)$ | Classification accuracy |
| **Class Potential** | 25 | $-\beta \log P(Y \mid K) + V_b$ | Class basin formation |
| **Behavior Cloning** | 25 | $-\log \pi(a^* \mid s)$ | Imitation learning |
| **Hierarchical** | 25 | $\sum_\ell \alpha_\ell \mathcal{L}^{(\ell)}$ | Multi-scale classification |
| **Governor Regret** | 26 | $\sum_t (\mathcal{L}_{\text{task}} + \gamma \text{ReLU}(C_k)^2)$ | Meta-learning objective |
| **Causal Info** | 29 | $\mathbb{E}[D_{\text{KL}}(p(\theta_W \mid z') \| p(\theta_W))]$ | Exploration via EIG |
| **Ontological Stress** | 33 | $(z_{\text{tex}}^{(\ell)})^i G^{(\ell)}_{ij} (z_{\text{tex}}^{(\ell)})^j$ | Texture predictability |
| **Sync Potential** | 37 | $\beta \int \mathcal{F}_{AB}^2 dA$ | Multi-agent alignment |
| **Joint Prediction** | 37 | $d_G(\hat{x}^A, x)^2 + d_G(\hat{x}^B, x)^2$ | Multi-agent world model |
| **Waste Quotient** | 38 | $1 - \Delta I / \int \dot{\mathcal{M}} dt$ | Consensus efficiency |
