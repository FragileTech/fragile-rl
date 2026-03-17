# Covariant Geometric World Model

## Overview

The world model predicts how the latent state of an embodied agent evolves over time, given a sequence of actions. It operates entirely within a **Poincaré ball** — a model of hyperbolic space where points live inside the open unit ball $\{z \in \mathbb{R}^D : \|z\| < 1\}$ — and all computations respect the Riemannian geometry of that space.

The core insight: the latent space is not flat. It has curvature, and the metric becomes singular at the boundary. A model that ignores this (e.g. concatenating $z$ and $a$ into an MLP) produces forces and predictions that do not transform correctly under changes of coordinates. Our model fixes this by using **CovariantAttention** — an attention mechanism with Christoffel-corrected queries, parallel-transported keys, and conformal temperature — as the fundamental building block for all learned force fields.

### What it does

Given:
- $z_0 \in \mathbb{B}^D$ — initial latent position (from the encoder)
- $a_{0:H-1} \in \mathbb{R}^{H \times A}$ — action sequence over a horizon of $H$ steps
- $w_0 \in \Delta^{K-1}$ — initial chart routing weights (soft assignment over $K$ atlas charts)

It produces a rollout:
$$z_0 \xrightarrow{a_0} z_1 \xrightarrow{a_1} z_2 \xrightarrow{a_2} \cdots \xrightarrow{a_{H-1}} z_H$$

along with chart predictions, momentum trajectories, and diagnostic quantities (jump rates, effective potential).

---

## The Geometry

### Poincaré ball metric

The Poincaré ball $\mathbb{B}^D$ has the conformal metric:

$$g_{ij}(z) = \lambda(z)^2 \, \delta_{ij}, \qquad \lambda(z) = \frac{2}{1 - \|z\|^2}$$

Key consequences:

| Quantity | Formula | Effect |
|---|---|---|
| **Conformal factor** $\lambda(z)$ | $2 / (1 - \|z\|^2)$ | Distances stretch near boundary |
| **Metric tensor** $G_{ij}$ | $\lambda^2 I$ | Diagonal, conformally flat |
| **Inverse metric** $G^{ij}$ | $\lambda^{-2} I$ | Lowers the "cost" of motion near center |
| **Christoffel symbols** $\Gamma^k_{ij} v^i v^j$ | $\frac{4(z \cdot v)v}{1 - \|z\|^2} - \frac{2\|v\|^2 z}{1 - \|z\|^2}$ | Geodesic acceleration correction |
| **Temperature** $\tau(z)$ | $\sqrt{d_k} / \lambda(z)$ | Sharper attention near boundary |

### Cotangent vs. tangent vectors

This distinction is critical:

- **Position** $z$ — a point in the ball
- **Velocity** $v = G^{-1} p$ — a **tangent (contravariant)** vector, used for geodesic drift
- **Momentum** $p$ — a **cotangent (covariant)** vector, the canonical conjugate variable
- **Force** $\partial \Phi / \partial z$ — a **cotangent** vector (Euclidean gradient), same type as $p$

Hamilton's equations connect them:

$$\dot{z}^i = G^{ij} p_j, \qquad \dot{p}_i = -\frac{\partial \Phi}{\partial z^i} + u_i^{\pi}$$

The momentum kick ($\dot{p}$) must be a covector. This means forces that modify $p$ are **Euclidean gradients** $\partial \Phi / \partial z$, not Riemannian gradients $G^{-1} \partial \Phi / \partial z$.

---

## Architecture

### Building blocks from `gauge.py`

All learned sub-modules use the same geometric primitives:

| Primitive | What it does |
|---|---|
| `CovariantAttention` | Cross-attention with `ChristoffelQuery` (encodes curvature via quadratic $\Gamma$ terms), `HyperbolicTransport` (parallel-transports keys into query frame via conformal factor ratios), and conformal temperature scaling |
| `ChristoffelQuery` | Query projection: $q = W_Q x + W_{Qz} z + z^T W_{Q\Gamma} z$ where the quadratic term encodes Christoffel corrections |
| `HyperbolicTransport` | Returns $\lambda_{\text{key}} / \lambda_{\text{query}}$ as a scalar scale factor per key (O(1) per key, not a matrix) |
| `ConformalMetric` | Computes $\lambda(z)$, $G(z)$, $G^{-1}(z)$, $\tau(z)$ |
| `SpectralLinear` | Lipschitz-bounded linear layers (spectral normalization) |
| `hyperbolic_distance` | Geodesic distance $d(x,y) = \frac{2}{\sqrt{c}} \text{artanh}(\sqrt{c}\, \|-x \oplus y\|)$ |
| `poincare_exp_map` | $\exp_z(v)$: moves from $z$ along tangent $v$ on the geodesic |
| `christoffel_contraction` | $\Gamma^k_{ij} v^i v^j$: geodesic acceleration correction (O(D), no matrices) |

### Tokenizers

Before feeding data to CovariantAttention, we need to create (feature, position) pairs:

**ActionTokenizer** — Lifts a flat action vector $a \in \mathbb{R}^A$ into $A$ context tokens:
- Each action component $a_i$ gets its own embedding: $x_i = a_i \cdot w_i + \text{pos}_i$, where $w_i \in \mathbb{R}^{d_\text{model}}$
- All tokens are positioned at the agent's current location: $z_i = z$ for all $i$
- Output: $(x, z) \in \mathbb{R}^{B \times A \times d_\text{model}} \times \mathbb{R}^{B \times A \times D}$

**ChartTokenizer** — Lifts chart routing weights $w \in \Delta^{K-1}$ into $K$ context tokens:
- Each chart $k$ has a learnable embedding $e_k \in \mathbb{R}^{d_\text{model}}$ weighted by $w_k$: $x_k = w_k \cdot e_k$
- Each chart has a learnable center $c_k \in \mathbb{B}^D$ (projected to stay inside the ball)
- Output: $(x, z) \in \mathbb{R}^{B \times K \times d_\text{model}} \times \mathbb{R}^{B \times K \times D}$

### Sub-modules

The world model has seven learned components. Each takes the agent's position $z$ and produces either a force, a scalar, or logits — all geometrically consistent.

#### 1. CovariantPotentialNet — "Where should the agent drift?"

Produces the **conservative force** (cotangent vector) that drives the Hamiltonian dynamics, plus a scalar potential for energy conservation monitoring.

**Force decomposition:**

$$F = \alpha \frac{\partial U}{\partial z} + (1 - \alpha)\, f_\text{critic}(z, K) + \gamma_\text{risk}\, f_\text{risk}(z, K)$$

- $U(z) = -2\,\text{artanh}(\|z\|)$ — **analytic** hyperbolic drive with exact gradient $\frac{\partial U}{\partial z} = \frac{-2z}{\|z\|(1 - \|z\|^2)}$ (no learnable parameters, no autograd)
- $f_\text{critic}$ — learned critic force via CovariantAttention over chart tokens
- $f_\text{risk}$ — learned risk force via CovariantAttention over chart tokens

**Scalar potential** (for diagnostics / energy conservation loss):

$$\Phi_\text{eff} = \alpha\, U(z) + (1 - \alpha)\, V_\text{critic} + \gamma_\text{risk}\, \Psi_\text{risk}$$

The critic and risk heads share attention features: each CovariantAttention computes features once, then two projection heads read off the force vector *and* the scalar simultaneously.

**Design decision — direct force prediction:** Early versions computed the scalar $\Phi(z)$ first and then differentiated with `torch.autograd.grad(create_graph=True)` to get the force. This required building second-order computation graphs — 1,050 times per epoch — and made training ~50x slower than necessary. The current design predicts forces directly, eliminating all `create_graph=True` calls. The analytical $\partial U / \partial z$ is exact, and the learned forces $f_\text{critic}, f_\text{risk}$ receive gradients through normal first-order backpropagation.

#### 2. CovariantControlField — "How does the action steer?"

Produces the **control force** $u^\pi(z, a, K) \in T^*_z \mathbb{B}^D$ (cotangent vector).

- Cross-attends from the agent's position to action tokens + chart tokens
- The concatenated context has $A + K$ tokens
- Output: tangent-space force vector via `SpectralLinear(d_model, D)`

#### 3. CovariantValueCurl — "Should the agent rotate?"

Produces an **antisymmetric field strength tensor** $\mathcal{F}_{ij}$ for the Boris rotation:

$$\mathcal{F} + \mathcal{F}^T = 0$$

- Cross-attends from $z$ to action tokens
- Outputs the $D(D-1)/2$ upper-triangular entries, then fills the antisymmetric matrix
- The antisymmetry guarantees the Boris rotation preserves $\|p\|$ exactly

#### 4. CovariantChartTarget — "Which chart should the agent be in next?"

Predicts chart transition logits $\ell \in \mathbb{R}^K$:

$$\ell_k = -\frac{d_\text{hyp}(z, c_k)}{\tau(z)} + z^T Q_{\Gamma}^{(k)} z + \text{action\_correction}$$

Three terms:
1. **Geodesic proximity**: negative hyperbolic distance to chart center $c_k$, temperature-scaled
2. **Christoffel correction**: quadratic term $z^T Q_\Gamma^{(k)} z$ capturing curvature effects
3. **Action correction**: mean-pooled action tokens projected to $K$ logits

Chart centers are `nn.Parameter`s projected inside the ball at every forward pass.

#### 5. CovariantJumpRate — "Should a chart transition happen?"

Predicts a **non-negative scalar** $\lambda(z, K) \geq 0$ (Poisson jump rate):

- Cross-attends from $z$ to chart tokens
- `softplus` output guarantees $\lambda \geq 0$
- Trained with an L1 sparsity penalty (jumps should be rare)

#### 6. CovariantMomentumInit — "What initial momentum?"

Initializes the momentum $p_0$ from the starting position:

$$p_0 = \lambda(z_0)^2 \cdot W z_0$$

The $\lambda^2$ factor ensures the initial momentum is a proper cotangent vector — metric-scaled so that $\|p\|$ has the right units relative to the local geometry.

#### 7. FactorizedJumpOperator — "Where to land after a jump?"

Transports the position between chart domains using Möbius addition. Shared with the encoder's atlas.

---

## The Integrator: Geodesic Boris-BAOAB

The dynamics are evolved using a **BAOAB splitting** — a symplectic integrator commonly used for Langevin dynamics — adapted for Riemannian geometry.

Each step of the horizon applies the following sequence:

### B — Momentum kick (first half)

```
force, _    = potential_net.force_and_potential(z, rw)    # cotangent force
u_pi        = control_net(z, action, rw)                  # control force
kick        = force - u_pi                                # net conservative force

p_minus     = p - (h/2) * kick                            # half kick
p_plus      = boris_rotation(p_minus, z, action)          # norm-preserving rotation
p           = p_plus - (h/2) * kick                       # complete first B step
```

The Boris rotation implements $p \mapsto p + (h\beta/2) G^{-1} \mathcal{F} p$ in a norm-preserving way (the Buneman-Boris trick).

### A — Geodesic drift (first half)

```
v           = G^{-1} p                      # raise index: cotangent → tangent
geo_corr    = christoffel_contraction(z, v) # Γ^k_ij v^i v^j
v_corr      = v - (h/4) * geo_corr         # corrected velocity
z           = exp_z((h/2) * v_corr)         # geodesic step via exponential map
z           = project_to_ball(z)            # safety: clamp ||z|| < 1
```

### O — Ornstein-Uhlenbeck thermostat

```
p = c₁ p + c₂ λ(z) ξ,    ξ ~ N(0, I)
```

where $c_1 = e^{-\gamma h}$ and $c_2 = \sqrt{(1 - c_1^2) T_c}$. The noise is scaled by the conformal factor $\lambda(z)$ so it respects the local geometry.

### A — Geodesic drift (second half)

Same as the first A step.

### B — Momentum kick (second half)

Same structure as the first B step, recomputed at the new position $z$.

### Jump process

After each BAOAB step, a **Poisson jump process** may trigger a discrete chart transition:

1. Compute jump rate $\lambda(z, w)$ and chart logits $\ell(z, a, w)$
2. Sample: $\text{jump} \sim \text{Bernoulli}(1 - e^{-\lambda \cdot dt})$
3. If jump: transport $z$ between source and target chart via `FactorizedJumpOperator`
4. Update chart weights: $w \leftarrow \text{softmax}(\ell)$

---

## Training: 3-Phase Curriculum

Training proceeds in three phases, each with its own parameter groups and loss functions.

### Phase 1 — Encoder warmup

**Goal:** Learn a good hyperbolic latent representation from single frames.

**Trainable:** Encoder + Decoder + JumpOperator

**Data:** Single frames from VLA feature cache (extracted from SmolVLA backbone).

**13 loss terms:**

| Loss | Weight | Purpose |
|---|---|---|
| Feature reconstruction (MSE) | `w_recon=1.0` | Reconstruct the input features |
| VQ commitment | `w_vq=1.0` | Vector quantization codebook |
| Routing entropy | `w_entropy=0.1` | Encourage sharp chart assignments |
| Enc-dec consistency (KL) | `w_consistency=0.1` | Encoder and decoder routing agree |
| Chart diversity | `w_diversity=0.1` | Prevent all data going to one chart |
| Hyperbolic uniformity | `w_uniformity=0.1` | Spread embeddings across the ball |
| Radial calibration | `w_radial_cal=0.1` | Calibrate radial distribution |
| Codebook spread | `w_codebook_spread=0.05` | Separate codebook vectors |
| Codebook centering | `w_codebook_center=0.01` | Prevent codebook drift |
| Chart collapse penalty | `w_chart_collapse=1.0` | Penalize dead charts |
| Code collapse penalty | `w_code_collapse=0.5` | Penalize dead codes |
| Window loss (info theory) | `w_window=0.5` | Mutual information bound on charts |
| Jump consistency | `w_jump=0.1` | Jump operator consistency (ramped) |

### Phase 2 — World model warmup

**Goal:** Train the world model to predict latent trajectories with the encoder frozen.

**Trainable:** World model only (encoder frozen)

**Data:** Sequences of $H$ frames with actions: `(features [B, H, D_feat], actions [B, H, A])`.

The encoder encodes all $H$ frames in a single batched pass (frozen, no grad). The world model then:
1. Takes $z_0$ and $w_0$ from the first frame
2. Rolls out $H-1$ steps using actions $a_{0:H-2}$
3. Compares predicted $\hat{z}_{1:H-1}$ and $\hat{k}_{1:H-1}$ against the encoder's outputs

**5 loss terms:**

| Loss | Weight | Formula | Purpose |
|---|---|---|---|
| Geodesic | `w_geodesic=1.0` | $\frac{1}{BH}\sum d_\text{hyp}(\hat{z}, z^*)$ | Position prediction accuracy |
| Chart transition | `w_chart_transition=0.5` | Cross-entropy on chart logits | Chart prediction accuracy |
| Momentum regularization | `w_momentum_reg=0.01` | $\frac{1}{2} p^T G^{-1} p$ (metric-aware kinetic energy) | Prevent momentum explosion |
| Energy conservation | `w_energy_conservation=0.01` | $\text{Var}_t[H(t)]$ across horizon | Integrator quality (symplecticity) |
| Jump dynamics | `w_jump_dynamics=0.1` | $\|\lambda\|_1$ (L1 on jump rates) | Sparsity: jumps should be rare |

The **geodesic loss** uses true hyperbolic distance (not MSE), so the loss landscape respects the geometry. Points near the boundary have naturally larger distances, which prevents the model from collapsing everything to the center.

The **momentum regularization** is metric-aware: kinetic energy is $\frac{1}{2} p^T G^{-1} p = \frac{1}{2} \left(\frac{1-\|z\|^2}{2}\right)^2 \|p\|^2$. This means the penalty is smaller near the boundary (where the inverse metric shrinks), correctly reflecting that the same Euclidean momentum corresponds to less "true" kinetic energy in regions of high curvature.

The **energy conservation loss** monitors Hamiltonian drift $H = \Phi_\text{eff} + T_\text{kinetic}$ across the horizon. A perfect symplectic integrator would keep $H$ constant; the variance penalty encourages the learned forces to be compatible with the integrator structure.

### Phase 3 — Joint fine-tuning

**Goal:** End-to-end fine-tuning of encoder + world model together.

**Trainable:** Everything (two param groups with different learning rates)

**Data:** Same sequence data as Phase 2.

**Loss:**

$$\mathcal{L}_\text{total} = \underbrace{0.1}_{\text{encoder scale}} \cdot \mathcal{L}_\text{encoder} + \underbrace{1.0}_{\text{dynamics scale}} \cdot \mathcal{L}_\text{dynamics}$$

- Encoder losses are the same 13 terms from Phase 1 (computed on frame 0)
- Dynamics losses are the same 5 terms from Phase 2
- The encoder LR is reduced (default: $10^{-4}$ vs $10^{-3}$ for the world model) to prevent catastrophic forgetting

The dynamics targets ($z^*$, $k^*$) are **detached** from the encoder to prevent the encoder from learning to produce "easy" targets instead of good representations.

---

## Hyperparameters

### Architecture

| Parameter | CLI flag | Default | Description |
|---|---|---|---|
| `latent_dim` | `--latent-dim` | 3 | Poincaré ball dimension |
| `num_charts` | `--num-charts` | 16 | Number of atlas charts |
| `d_model` | `--wm-d-model` | 128 | CovariantAttention feature width |
| `action_dim` | `--action-dim` | 6 | Action space dimension (SO100: 6 joints) |
| `hidden_dim` | `--hidden-dim` | 256 | Encoder hidden dimension |
| `codes_per_chart` | `--codes-per-chart` | 64 | VQ codebook size per chart |

### Integration

| Parameter | CLI flag | Default | Description |
|---|---|---|---|
| `dt` | `--wm-dt` | 0.01 | Integration time step |
| `gamma_friction` | `--wm-gamma-friction` | 1.0 | Langevin friction coefficient |
| `T_c` | `--wm-T-c` | 0.1 | Thermostat temperature |
| `alpha_potential` | `--wm-alpha-potential` | 0.5 | Balance: analytic drive vs learned critic |
| `beta_curl` | `--wm-beta-curl` | 0.1 | Value-curl coupling (Boris rotation strength) |
| `gamma_risk` | `--wm-gamma-risk` | 0.01 | Risk penalty weight |
| `use_boris` | `--wm-use-boris` / `--wm-no-boris` | True | Enable Boris rotation |
| `use_jump` | `--wm-use-jump` / `--wm-no-jump` | True | Enable Poisson jump process |

### Training

| Parameter | CLI flag | Default | Description |
|---|---|---|---|
| Phase 1 epochs | `--phase1-epochs` | 100 | Encoder warmup epochs (0 to skip) |
| Phase 2 epochs | `--phase2-epochs` | 50 | World model warmup epochs |
| Phase 3 epochs | `--phase3-epochs` | 50 | Joint fine-tuning epochs |
| LR (encoder) | `--lr` | $10^{-3}$ | Phase 1 encoder learning rate |
| LR (world model) | `--lr-wm` | $10^{-3}$ | Phase 2 world model learning rate |
| LR (joint encoder) | `--lr-joint-encoder` | $10^{-4}$ | Phase 3 encoder learning rate |
| LR (joint WM) | `--lr-joint-wm` | $10^{-3}$ | Phase 3 world model learning rate |
| Batch size | `--batch-size` | 256 | Training batch size |
| Sequence length | `--sequence-length` | 8 | Temporal horizon for sequences |
| Gradient clipping | `--grad-clip` | 1.0 | Max gradient norm |

---

## Running

### Feature extraction (one-time)

```bash
uv run fragile vla-extract -- --max-episodes 0
```

### Training

Minimal (world model only, skip encoder warmup):

```bash
uv run fragile vla-joint -- \
  --phase1-epochs 0 --phase2-epochs 0 --phase3-epochs 500 \
  --latent-dim 3 --num-charts 16 --codes-per-chart 64 \
  --wm-d-model 128 --log-every 1 --save-every 20 --device cuda
```

Full 3-phase:

```bash
uv run fragile vla-joint -- \
  --phase1-epochs 100 --phase2-epochs 50 --phase3-epochs 50 \
  --latent-dim 16 --num-charts 8 --codes-per-chart 32 \
  --hidden-dim 256 --wm-d-model 128 \
  --log-every 5 --save-every 50 --device cuda
```

Resume from checkpoint:

```bash
uv run fragile vla-joint -- --resume outputs/vla/joint/p2_epoch_00049.pt \
  --phase1-epochs 0 --phase2-epochs 0 --phase3-epochs 100 --device cuda
```

---

## File map

| File | Role |
|---|---|
| `src/fragile/learning/vla/covariant_world_model.py` | All sub-modules + `GeometricWorldModel` class |
| `src/fragile/learning/vla/losses.py` | All loss functions (5 dynamics + 8 encoder helpers + 3 phase assemblers) |
| `src/fragile/learning/vla/train_joint.py` | 3-phase training loop, CLI, checkpointing |
| `src/fragile/learning/vla/config.py` | `VLAConfig` dataclass (all defaults) |
| `src/fragile/learning/core/layers/gauge.py` | Hyperbolic geometry primitives + CovariantAttention |
| `src/fragile/learning/core/layers/atlas.py` | Chart router, `_project_to_ball` |
| `src/fragile/learning/core/layers/topology.py` | `FactorizedJumpOperator`, jump consistency |
| `src/fragile/learning/core/layers/primitives.py` | `SpectralLinear`, `NormGatedGELU` |
| `tests/fractalai/learning/test_covariant_world_model.py` | 30 tests covering shapes, geometry, gradients |

---

## Key design decisions

1. **Direct force prediction over autograd.** Computing $\partial \Phi / \partial z$ via `torch.autograd.grad(create_graph=True)` required second-order computation graphs on every BAOAB step (~1,050 per epoch). Switching to direct force prediction with an analytical $\partial U / \partial z$ eliminated this entirely, giving a ~50x speedup.

2. **Scalar transport, not matrix transport.** The parallel transport on the Poincaré ball is $P_{x \to y}(v) = (\lambda_x / \lambda_y) v$ — a scalar multiplication, not a matrix. `HyperbolicTransport` returns `[B, N, 1]` instead of `[B, N, d_k, d_k]`, reducing memory by ~160,000x per attention call.

3. **Chart centers projected every forward pass.** `nn.Parameter` chart centers can drift outside $\|z\| < 1$ during gradient updates. We apply `_project_to_ball` at every forward pass so the geometry stays valid.

4. **SpectralLinear everywhere.** All linear layers use spectral normalization for Lipschitz-bounded weights. This prevents the learned forces from growing too large and destabilizing the integrator.

5. **Detached dynamics targets in Phase 3.** The encoder targets $z^*$ for the world model are detached during joint training. Otherwise the encoder learns to produce embeddings that are easy to predict rather than informative.

6. **Metric-aware losses.** The momentum regularization uses proper kinetic energy $\frac{1}{2} p^T G^{-1} p$, not Euclidean $\frac{1}{2}\|p\|^2$. The geodesic loss uses hyperbolic distance, not MSE. These ensure the loss landscape matches the geometry of the latent space.
