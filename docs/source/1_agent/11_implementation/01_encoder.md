(sec-shared-dynamics-encoder)=
# The Shared-Dynamics Encoder: From Theory to Implementation

## TLDR

- The **shared-dynamics encoder** eliminates the separate dynamics codebook by routing both
  reconstruction and Markov-transition prediction through the *same* VQ codebook.
- This forces the codebook to learn symbols that are simultaneously good for decoding *and*
  for predicting the next state --- exactly the causal enclosure condition
  ({prf:ref}`def-causal-enclosure`) demands.
- Training is **unified from epoch 0**: reconstruction, VQ, and dynamics losses all flow
  through one forward pass, one backward pass, one optimizer step.
- The encoding pipeline maps observations to the Poincare ball, assigns charts via covariant
  attention, computes chart-local coordinates via Mobius addition, and quantizes against a
  hyperbolic codebook.
- Diagnostics track symbol usage entropy, per-chart code utilization, and the Zeno smoothness
  penalty to verify that the learned symbols satisfy macro closure in practice.

## Roadmap

1. Why a shared codebook? The motivation from causal enclosure.
2. The full encoding pipeline: pixels to symbols, step by step.
3. Hyperbolic vector quantization: VQ in the Poincare ball.
4. The three-channel decomposition as it appears in code.
5. The shared codebook override: what `SharedDynAtlasEncoder` changes and why.
6. Markov dynamics on the symbol space: transition model and Zeno loss.
7. The unified training loop: everything at once.
8. Diagnostics: knowing your encoder is working.
9. Summary: theory-to-code correspondence table.

(sec-why-shared-codebook)=
## Why a Shared Codebook?

:::{div} feynman-prose
Let me tell you about a mistake we made first, because the mistake is instructive.

The original pipeline had three phases. Phase 1: train the encoder to reconstruct observations.
This gives you a nice codebook --- symbols that capture what things *look like*. Phase 2: freeze
the encoder, bolt on a separate dynamics codebook, and train it to predict transitions. Phase 3:
fine-tune everything jointly.

This works, but it is wasteful in a way that should bother you. Think about it. In Phase 1 the
codebook learns "these pixels go with code 7." In Phase 2 a *different* codebook learns "code 3
transitions to code 5 under action $a$." The two codebooks are speaking different languages. Phase 3
is supposed to reconcile them, but in practice it amounts to duct-taping two systems together and
hoping the gradients sort things out.

Here is the key insight. The causal enclosure condition ({prf:ref}`def-causal-enclosure`) says that
the macro symbol $K_t$ must predict $K_{t+1}$ given the action. The reconstruction loss says that
$K_t$ must be informative enough to decode the observation. These are not two separate requirements
on two separate codebooks. They are two requirements on the *same* symbol. If the symbol is any good,
it should satisfy both.

So why not use one codebook and demand both? That is exactly what the shared-dynamics encoder does.
One codebook. Two loss signals. From the very first epoch.
:::

The theoretical motivation comes from two places:

1. **Causal enclosure** ({prf:ref}`def-causal-enclosure`): the macro state
   $K_t = (K_{\mathrm{chart}}, K_{\mathrm{code}})$ must support a concentrated transition kernel
   $P(K_{t+1} \mid K_t, a_t)$. If the codebook symbols are not predictive, this kernel is diffuse
   and the closure defect ({prf:ref}`def-closure-defect`) is large.

2. **Conditional independence** ({ref}`sec-conditional-independence-and-sufficiency`): texture
   $z_{\mathrm{tex}}$ must be independent of the next macro state given $K_t$ and $a_t$. A separate
   dynamics codebook can "cheat" by encoding dynamics information that the reconstruction codebook
   missed, rather than forcing the shared representation to be sufficient.

:::{admonition} Researcher Bridge: VQ-VAE Codebook Sharing
:class: info
In standard VQ-VAE ({ref}`sec-the-shutter-as-a-vq-vae`), the codebook is trained by reconstruction
alone. Adding a dynamics objective to the *same* codebook is analogous to the "world model" losses in
Dreamer-v3 or IRIS, but operating on discrete symbols in hyperbolic space rather than continuous
latents in Euclidean space. The shared codebook ensures the discrete bottleneck does not discard
dynamically relevant information.
:::


(sec-encoding-pipeline)=
## The Encoding Pipeline --- From Pixels to Symbols

:::{div} feynman-prose
Let me walk you through what actually happens when an observation enters the encoder. I want you to
have a concrete picture in your head, not just a vague sense of "it gets encoded."

Imagine a postal system. A letter (the observation) arrives at the central sorting office. First, the
office reads the address and figures out which region the letter belongs to --- that is the chart
assignment. Then within that region, it finds the nearest post office --- that is the VQ code. The
difference between the exact address and the post office location is the nuisance: orientation of the
house, color of the door, things that matter for delivery but not for routing.

The twist is that this postal system operates in hyperbolic space, where distances behave differently
near the boundary than near the center. That is not just mathematical affectation. It gives the
system a natural hierarchy: points near the center of the Poincare ball are "generic" and points near
the edge are "specific." The chart centers sit at moderate radius, and codes within each chart fan
out from there.
:::

The full forward pass through `PrimitiveAttentiveAtlasEncoder` proceeds in five stages.

```{mermaid}
flowchart LR
    X["x [B, D_in]"] --> FE["feature_extractor"]
    FE --> VP["val_proj + scale"]
    VP --> STB["smooth_tangent_to_ball"]
    STB --> V["v [B, D]<br/>(Poincare ball)"]
    V --> CR["CovariantChartRouter"]
    CC["chart_centers [N_c, D]"] --> CR
    CR --> RW["router_weights [B, N_c]"]
    CR --> KC["K_chart [B]"]
    RW --> CBAR["Mobius barycenter"]
    CC --> CBAR
    CBAR --> CB["c_bar [B, D]"]
    V --> ML["v_local = Mobius(-c_bar, v)"]
    CB --> ML
    ML --> VL["v_local [B, D]"]
    VL --> HVQ["_hyperbolic_vq"]
    HVQ --> ZQ["z_q_blended [B, D]"]
    HVQ --> KCD["K_code [B]"]
    VL --> SF["structure_filter"]
    ZQ --> SF
    SF --> ZN["z_n [B, D]"]
    SF --> ZTEX["z_tex [B, D]"]
    CB --> ZGEO["z_geo (Mobius reassembly)"]
    ZQ --> ZGEO
    ZN --> ZGEO
```

### Feature Extraction and Poincare Mapping

The first two stages are straightforward: a feed-forward network extracts features, and a linear
projection maps them into the latent dimension. The result is then smoothly mapped into the Poincare
ball.

```python
# PrimitiveAttentiveAtlasEncoder.forward (atlas.py)
features = self._encode_features(x)                     # [B, hidden_dim]
v_raw = self.val_proj(features) * self.val_proj_scale    # [B, latent_dim]
v = _smooth_tangent_to_ball(v_raw)                       # [B, D], inside Poincare ball
```

The function `_smooth_tangent_to_ball` applies a tanh-based capping in the tangent space at the
origin, then maps via the exponential map. This ensures `v` always lies strictly inside the ball
($\|v\| < 1$) without hard clipping artifacts.

:::{note}
:class: feynman-added
The learnable `val_proj_scale` parameter controls how aggressively points spread toward the boundary
of the Poincare ball. Early in training a large scale pushes points outward, increasing chart
separation. As the encoder converges, gradients can reduce this scale if the geometry demands it.
:::

### Chart Routing via Covariant Attention

The `CovariantChartRouter` ({ref}`sec-covariant-cross-attention-architecture`) computes a soft
distribution over charts using hyperbolic-distance-based scores with three ingredients:

1. **Hyperbolic distance scoring:** $\text{score}(v, c_i) = -d_H(v, c_i) / \tau(v)$, where $d_H$
   is the Poincare distance and $\tau(v) = \sqrt{D}(1 - \|v\|^2)/2$ is a position-dependent
   temperature.

2. **Christoffel correction:** A quadratic $\Gamma$-term captures how the metric varies across the
   ball, implemented as a tensorized product $z^\top \gamma z$ (full or low-rank).

3. **Parallel transport of queries:** Chart query vectors are transported from the origin to the
   observation point $v$ using the conformal factor $\lambda(z) = 2/(1 - \|z\|^2)$, giving an
   $O(n)$ alternative to the $O(n^3)$ Cayley transform.

```python
# CovariantChartRouter.forward produces:
router_weights  # [B, N_c] — soft (or Gumbel-hard) chart probabilities
K_chart         # [B]     — hard chart assignment (argmax)
```

:::{div} feynman-prose
Why all this machinery for routing? Because we are doing geometry on a curved space, and getting the
routing right is crucial. In flat space you can just take dot products and call it a day. But in the
Poincare ball, a point near the boundary is *far* from everything in the interior --- the hyperbolic
distance diverges logarithmically as you approach the edge. If you use naive Euclidean dot products,
the router cannot distinguish between "this point is in chart 3" and "this point is on the boundary
and equidistant from everything."

The position-dependent temperature $\tau(v)$ handles this: near the center the temperature is high
(uncertain routing), near the boundary it is low (sharp routing). This is not a tunable hyperparameter
--- it falls out of the geometry of the Poincare metric. And that is how it should be: the geometry
tells you how confident you can be about chart assignment at each point.

For the full theoretical justification, see the attentive atlas
({ref}`sec-tier-the-attentive-atlas`) and the WFR geometry
({ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`).
:::

### Chart-Local Coordinates via Mobius Transport

Once we know the chart mixture weights, we compute the weighted barycenter $\bar{c}$ and transport
$v$ into chart-local coordinates:

```python
c_bar = _poincare_weighted_mean(chart_centers, router_weights)  # [B, D]
v_local = _project_to_ball(mobius_add(-c_bar, v))               # [B, D]
```

The Mobius addition $(-\bar{c}) \oplus v$ is the Poincare-ball analogue of translation: it moves the
chart center to the origin, carrying $v$ along for the ride. In chart-local coordinates, all charts
"look the same" near their center --- this is the gauge covariance that makes the codebook
chart-independent.

:::{note}
:class: feynman-added
The `_project_to_ball` call after Mobius addition is a safety projection that clamps the norm to
$< 0.99$. Numerical errors in hyperbolic arithmetic can occasionally push points outside the ball,
and the projection prevents `NaN` cascades in subsequent operations.
:::


(sec-hyperbolic-vq)=
## Hyperbolic Vector Quantization

:::{div} feynman-prose
Now we arrive at the heart of the encoder: vector quantization in the Poincare ball.

Standard VQ-VAE works in Euclidean space. You have a codebook of vectors, you find the nearest one
by $L^2$ distance, and you snap to it. Simple. But we are working in hyperbolic space, so "nearest"
means nearest in the *hyperbolic metric*. The codebook lives in the Poincare ball, the observations
live in the Poincare ball, and distances are computed via the Mobius gyrogroup.

Why bother? Because the hyperbolic metric gives you exponential volume growth with radius. A
codebook of $K$ codes in hyperbolic space can tile the ball with much finer resolution near the
boundary than near the center. This is exactly what you want for a hierarchical representation:
coarse distinctions near the center, fine distinctions near the edge. See
{ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits` for how this connects to
the information-theoretic capacity constraint.

Here is how `_hyperbolic_vq` works, step by step.
:::

### The Algorithm

The `_hyperbolic_vq` method in `PrimitiveAttentiveAtlasEncoder` (atlas.py) takes chart-local
coordinates `v_local` and quantizes against a codebook parameter:

**Step 1. Project codebook to ball.**

```python
codebook = _project_to_ball(codebook_param)  # [N_c, K, D]
```

**Step 2. Compute Mobius differences.**

```python
diff = mobius_add(-codebook_exp, v_exp)       # [B, N_c, K, D]
diff_tan = log_map_zero(diff)                 # map to tangent space at origin
```

The Mobius difference $(-c_k) \oplus v$ measures "how far is $v$ from codebook entry $c_k$" in the
hyperbolic sense. Mapping to the tangent space via `log_map_zero` linearizes this so we can use
ordinary $L^2$ norms.

**Step 3. Find nearest code per chart.**

```python
dist = (diff_tan ** 2).sum(dim=-1)            # [B, N_c, K]
indices = torch.argmin(dist, dim=-1)          # [B, N_c]
```

**Step 4. Gather nearest codes and blend via router weights.**

```python
z_q_blended = _poincare_weighted_mean(z_q_all, router_weights)  # [B, D]
K_code = indices.gather(1, K_chart.unsqueeze(1)).squeeze(1)      # [B]
```

The blending uses a hyperbolic barycenter (log-map at origin, weighted average, exp-map back),
not Euclidean interpolation.

### Commitment and Codebook Losses

The VQ loss has two terms, both computed in tangent space and weighted by the router probabilities:

$$
\mathcal{L}_{\mathrm{VQ}} = w_{\mathrm{cb}} \sum_c w_c \| \log_0((-v) \oplus z_q^c) \|^2
    + \beta \sum_c w_c \| \log_0((-z_q^c) \oplus v) \|^2
$$

- **Codebook loss** (weight $w_{\mathrm{cb}}$): pushes codebook entries toward the encoder outputs
  (gradients flow only into the codebook).
- **Commitment loss** (weight $\beta$): pushes encoder outputs toward the codebook entries
  (gradients flow only into the encoder).

The router weights $w_c$ ensure that only the "active" chart's codes receive strong gradient signal.

:::{div} feynman-prose
The commitment loss deserves a moment of thought. Without it, the encoder is free to move its outputs
anywhere --- the codebook chases the encoder, the encoder runs away, and nothing converges. The
commitment term says: "encoder, you have chosen code 7. Now *commit* to it. Do not drift away
capriciously." The $\beta$ parameter controls how strongly we enforce this. Too high, and the encoder
is frozen in place. Too low, and it wanders. In practice $\beta = 0.25$ works well.

For the information-theoretic view of why this bottleneck is necessary, see
{ref}`sec-causal-information-bound`.
:::

### The Straight-Through Estimator

VQ involves a discrete `argmin`, which has zero gradient everywhere. The straight-through estimator
bypasses this: in the forward pass we use the quantized code $z_q$, but in the backward pass we
pass gradients through as if $z_q = v_{\mathrm{local}}$. This is implemented explicitly in the
$z_{\mathrm{geo}}$ assembly (see the next section): `delta_to_code.detach()` cuts the gradient
through the VQ selection while preserving the forward value, so the decoder's reconstruction loss
flows gradients directly into the encoder's `v_local`.


(sec-three-channel-code)=
## The Three-Channel Decomposition in Code

:::{div} feynman-prose
Now here is where the theory ({prf:ref}`def-three-channel-latent`) meets the code.

We have the quantized code $z_q$ --- that is the macro symbol. But there is information in the
observation that the discrete code does not capture. Where does it go? Into two residual channels:
nuisance $z_n$ and texture $z_{\mathrm{tex}}$.

Think of it this way. The code $z_q$ is like the name of a city: "San Francisco." The nuisance
$z_n$ is the neighborhood and street --- structured information that varies continuously but does
not change the city label. The texture $z_{\mathrm{tex}}$ is the color of the paint on your house
--- it matters for reconstruction (you want to draw the picture correctly) but has no bearing on
dynamics (your house color does not affect the weather tomorrow).
:::

After VQ, the encoder computes the residual between the input and the quantized code, then splits
it through a learned `structure_filter`:

```python
# Broadcast v_local for per-chart residuals
v_bc = v_local.unsqueeze(1)                                        # [B, 1, D]

# Residual in tangent space (per chart, broadcast over N_c)
delta = log_map_zero(mobius_add(-z_q_all.detach(), v_bc))          # [B, N_c, D]

# Structure filter extracts nuisance
z_n_all = self.structure_filter(delta)                              # [B, N_c, D]

# Blend nuisance across charts, then compute texture as the remainder
z_n_tan = (z_n_all * router_weights.unsqueeze(-1)).sum(dim=1)      # [B, D]
z_tex = delta_blended - z_n_tan                                     # [B, D]
```

The `structure_filter` is a gauge-covariant two-layer network (`IsotropicBlock` followed by
`SpectralLinear`). The `IsotropicBlock` applies a rotation-equivariant nonlinearity, and the
`SpectralLinear` uses spectral normalization for Lipschitz control. Together they learn to
separate dynamically-relevant residual (nuisance) from dynamically-irrelevant residual (texture).
The enclosure loss ({prf:ref}`def-causal-enclosure-condition`) provides the training signal that
enforces this split: if texture leaks dynamics information, the enclosure probe detects it and
pushes gradients back through the structure filter.

The geometric latent for the decoder reassembles all three components via a multi-step Mobius
reconstruction:

```python
# 1. Tangent-space displacement from v_local to quantized code
delta_to_code = log_map_zero(mobius_add(-v_local, z_q_blended))       # [B, D]

# 2. Straight-through reattachment (detach stops VQ gradient, keeps encoder gradient)
z_q_st = mobius_add(v_local, exp_map_zero(delta_to_code.detach()))    # [B, D]

# 3. Add nuisance in tangent space, then lift back to ball
z_local = mobius_add(z_q_st, exp_map_zero(z_n_tan))                  # [B, D]

# 4. Translate from chart-local back to global coordinates
z_geo = _project_to_ball(mobius_add(c_bar, z_local))                  # [B, D]
```

Step 2 is the straight-through estimator: `delta_to_code.detach()` means the forward pass snaps to
the nearest code, but the backward pass treats $z_{q,\mathrm{st}}$ as if it were $v_{\mathrm{local}}$,
allowing gradients to flow through the encoder. The final `_project_to_ball` clamps the result
inside the ball to prevent numerical drift.


(sec-shared-codebook)=
## The Shared Codebook --- Dynamics Meets Reconstruction

:::{div} feynman-prose
And here is the beautiful thing. All that machinery I just described --- the feature extraction,
the chart routing, the hyperbolic VQ, the three-channel split --- that is the standard
`PrimitiveAttentiveAtlasEncoder`. The shared-dynamics encoder changes exactly *one* method.

One method. That is all.

The insight is that the parent class has a `dynamics_vq` method that quantizes against a *separate*
dynamics codebook (`self.codebook_dyn`). The `SharedDynAtlasEncoder` overrides this to quantize
against *the same* codebook (`self.codebook`) that reconstruction uses. And it sets the VQ loss
weights to zero, because the main forward pass already trains the codebook.

That is the entire change. Let me show you.
:::

### `SharedDynAtlasEncoder`: What It Changes

The class inherits from `PrimitiveAttentiveAtlasEncoder` and makes two modifications:

```python
class SharedDynAtlasEncoder(PrimitiveAttentiveAtlasEncoder):

    def __init__(self, **kwargs):
        kwargs["dyn_codes_per_chart"] = 0   # no separate codebook_dyn parameter
        super().__init__(**kwargs)

    def dynamics_vq(self, v_local, router_weights):
        return self._hyperbolic_vq(
            v_local,
            self.codebook,        # <-- same codebook as reconstruction
            router_weights,
            0.0,                  # commitment_beta = 0
            0.0,                  # codebook_loss_weight = 0
            use_soft_equiv=False,
        )[:4]
```

**Why zero VQ loss?** Because the reconstruction forward pass already computes
$\mathcal{L}_{\mathrm{VQ}}$ against `self.codebook` with the configured $\beta$ and codebook loss
weight. If the dynamics path *also* applied VQ losses, the codebook would receive double gradients
--- once from reconstruction, once from dynamics --- with no principled way to balance them. Setting
the dynamics VQ loss to zero means: "use the codebook for quantization, but let the reconstruction
loss be the sole codebook trainer."

The dynamics *transition* loss still flows gradients into the codebook *indirectly*. The transition
model receives the continuous VQ output (`code_features=z_q_dyn_all[:, t]`) rather than only the
discrete code index. Since `torch.gather` (which retrieves codebook entries by index) is
differentiable with respect to the gathered values, the transition cross-entropy loss backpropagates
through the MLP, through the gathered features, and into the codebook entries that were selected.
This provides a gentle pressure for the codebook to organize itself in a way that makes transitions
predictable, without the instability of competing VQ objectives.

### `SharedDynTopoEncoder`: The Wrapper

The `SharedDynTopoEncoder` is the user-facing class. It inherits from `TopoEncoderPrimitives`
({ref}`sec-topoencoder-architecture`), which builds both an encoder and a decoder. After
`super().__init__()`, it replaces `self.encoder` with a `SharedDynAtlasEncoder`:

```python
class SharedDynTopoEncoder(TopoEncoderPrimitives):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Replace encoder; decoder is untouched
        self.encoder = SharedDynAtlasEncoder(**encoder_kwargs)
```

The decoder never sees the change. It receives `z_geo` exactly as before. The only difference is
that the codebook entries are now shaped by dynamics pressure as well as reconstruction pressure.

:::{admonition} Researcher Bridge: Shared vs. Separate Codebooks
:class: info
The choice between shared and separate codebooks is a bias-variance tradeoff. A separate dynamics
codebook has more capacity (it can represent transitions that the reconstruction codebook misses)
but less inductive bias (nothing forces the two to agree). The shared codebook has less capacity
but stronger inductive bias: the reconstruction bottleneck *is* the dynamics bottleneck. The
causal enclosure theory ({prf:ref}`def-causal-enclosure`) says this is the right bias --- the
macro symbol should be sufficient for both tasks by construction.
:::


(sec-markov-dynamics-symbols)=
## Markov Dynamics on the Symbol Space

:::{div} feynman-prose
Now that the encoder produces macro symbols, we need to predict how those symbols evolve over time.
This is where the `DynamicsTransitionModel` comes in.

The model is deliberately simple: an MLP that takes the current chart embedding $\bar{c}_t$, the
current dynamics code embedding, and the action $a_t$, and predicts logits over the joint
$(K_{\mathrm{chart}}, K_{\mathrm{code}})$ space at $t+1$. That is a flat softmax over
$N_c \times K_{\mathrm{codes}}$ states.

Why so simple? Because the whole point of the macro abstraction is that dynamics at this level
*should be* simple. If you need a transformer with 12 heads to predict the next macro state, your
macro states are not abstract enough. The simplicity of the transition model is a feature, not a
limitation --- it puts pressure on the encoder to find symbols where a simple predictor suffices.
See {ref}`sec-the-equations-of-motion-geodesic-jump-diffusion` for the theoretical underpinning.
:::

### The Transition Model

`DynamicsTransitionModel` (in `losses.py`) predicts the next joint state:

$$
P(K_{t+1} \mid \bar{c}_t, K_{\mathrm{code},t}, a_t)
$$

where $K_{t+1} = (K_{\mathrm{chart},t+1}, K_{\mathrm{code},t+1})$ is flattened to a single index
in $\{0, \ldots, N_c \cdot K_{\mathrm{codes}} - 1\}$.

```python
class DynamicsTransitionModel(nn.Module):
    def __init__(self, chart_dim, action_dim, num_charts, dyn_codes_per_chart, hidden_dim=128):
        self.num_states = num_charts * dyn_codes_per_chart
        self.code_embed = nn.Embedding(dyn_codes_per_chart, chart_dim)
        self.mlp = nn.Sequential(
            nn.Linear(chart_dim + chart_dim + action_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_states),
        )

    def forward(self, chart_embed, action, code_idx=None, code_features=None):
        # Returns logits [B, num_states]
        if code_features is not None:
            code_e = code_features
        else:
            code_e = self.code_embed(code_idx)
        inp = torch.cat([chart_embed, code_e, action], dim=-1)
        return self.mlp(inp)
```

The input is the concatenation of the chart embedding (the hyperbolic barycenter $\bar{c}_t$), the
code embedding (looked up from a learned embedding table or provided directly), and the action.

### The Markov Transition Loss

The `compute_dynamics_markov_loss` function loops over the temporal horizon and computes:

1. **Dynamics VQ at each timestep:** `atlas_encoder.dynamics_vq(v_local_t, router_weights_t)` ---
   in the shared-codebook variant, this quantizes against the reconstruction codebook with zero VQ
   loss.

2. **Transition cross-entropy:** for each consecutive pair $(t, t+1)$, the transition model
   predicts logits over the next state and we compute
   $\mathrm{CE}(\text{logits}, K_{\mathrm{chart},t+1} \cdot K_{\mathrm{codes}} + K_{\mathrm{code},t+1})$.

3. **Transition accuracy:** the fraction of next-state predictions that match the ground truth
   (for monitoring, not for the loss).

$$
\mathcal{L}_{\mathrm{dyn}} = \mathcal{L}_{\mathrm{VQ,dyn}} + w_{\mathrm{trans}} \cdot
  \frac{1}{H-1} \sum_{t=0}^{H-2} \mathrm{CE}\bigl(\hat{P}(K_{t+1}), K_{t+1}^{\mathrm{true}}\bigr)
  + w_{\mathrm{Zeno}} \cdot \mathcal{L}_{\mathrm{Zeno}}
$$

In the shared-codebook setting, $\mathcal{L}_{\mathrm{VQ,dyn}} = 0$ because the dynamics VQ uses
zero commitment and codebook loss weights. The Zeno term is described below.

### Zeno Smoothness

:::{div} feynman-prose
There is one more loss term that deserves attention: the Zeno penalty.

The name comes from Zeno's paradox. If the encoder assigns chart 1 at time $t$, chart 3 at $t+1$,
chart 1 at $t+2$, chart 3 at $t+3$ --- rapidly flipping back and forth --- something has gone
wrong. Physical systems do not teleport between distant states at every timestep. This flickering is
a sign that the chart boundaries are poorly placed, or that the router is uncertain and breaking ties
differently on near-identical inputs.

The Zeno loss penalizes this by measuring the Jensen-Shannon divergence (or KL divergence, depending
on configuration) between consecutive *transition prediction* distributions. Concretely, at each
timestep the transition model outputs a softmax over the joint state space; the Zeno term compares
these softmax distributions at adjacent steps:

$$
\mathcal{L}_{\mathrm{Zeno}} = \frac{1}{H-2} \sum_{t=1}^{H-2}
  \mathrm{JSD}\bigl(\hat{P}(K_{t+1} \mid s_t, a_t) \;\|\; \hat{P}(K_{t+2} \mid s_{t+1}, a_{t+1})\bigr)
$$

This pushes the transition model's predictions to change smoothly over time. If the predicted
next-state distribution wildly oscillates between consecutive steps, the Zeno penalty fires.
Gradients flow back through the transition model and (via the straight-through estimator) into
the encoder, encouraging symbol assignments that yield smooth dynamics.

For how this connects to the theoretical equations of motion, see
{ref}`sec-the-equations-of-motion-geodesic-jump-diffusion`. For the filtering interpretation, see
{ref}`sec-filtering-template-on-the-discrete-macro-register`.
:::


(sec-unified-training)=
## Unified Training --- Everything at Once

:::{div} feynman-prose
Here is the practical payoff of the shared codebook: the training loop becomes simple.

No phases. No freezing. No "train the encoder for 50 epochs, then switch to dynamics, then
fine-tune." One loop, one optimizer, one loss. From epoch 0, the codebook hears from both the
reconstruction gradient and the dynamics gradient. It learns symbols that serve both masters
simultaneously.

Let me walk you through one training step.
:::

The training loop in `shared_dyn/train.py` (`_run_unified`) processes mini-batches of
observation sequences $\{x_0, x_1, \ldots, x_{H-1}\}$ with associated actions
$\{a_0, a_1, \ldots, a_{H-1}\}$.

### One Training Step

**Step 1. Reconstruction on frame 0.**

```python
(base_loss, zn_reg_loss, metrics, ...) = _compute_encoder_losses(
    features[:, 0, :], model, jump_op, args, epoch,
    hard_routing=current_hard_routing,
    hard_routing_tau=current_tau,
    phase1_config=phase1_config,
)
```

This computes the full encoder loss on the first frame: reconstruction MSE, VQ loss, routing
entropy, consistency, chart usage, codebook spread, and all the regularizers documented in
{ref}`sec-loss-function-enforcing-macro-micro-separation`.

**Step 2. Encode remaining frames (no reconstruction).**

```python
(K_rest, ..., v_local_rest, _) = model.encoder(
    rest, hard_routing=..., hard_routing_tau=...,
)
```

Frames $1$ through $H-1$ are encoded but not decoded. We need their chart assignments,
router weights, and chart-local coordinates for the dynamics loss, but we do not need to
reconstruct them. This saves compute.

**Step 3. Dynamics Markov loss over the full horizon.**

```python
L_dyn, dyn_metrics, _ = compute_dynamics_markov_loss(
    model.encoder, dyn_trans_model,
    v_local_all, rw_all, c_bar_all, K_all, actions,
    transition_weight=args.w_dyn_transition,
    zeno_weight=args.w_zeno,
)
```

This calls `dynamics_vq` (which, in the shared variant, uses the reconstruction codebook) at each
timestep, then computes the transition CE and Zeno smoothness over the sequence.

**Step 4. Total loss and backward pass.**

$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{base}} + \mathcal{L}_{\mathrm{zn\text{-}reg}}
  + \mathcal{L}_{\mathrm{dyn}}
$$

```python
total = base_loss + zn_reg_loss + L_dyn
optimizer.zero_grad()
total.backward()
torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
optimizer.step()
```

### Loss Composition

:::{div} feynman-added
| Loss Term | Source | What It Trains | Typical Weight |
|-----------|--------|----------------|----------------|
| Reconstruction MSE | Frame 0 decoder output | Encoder + decoder | `w_recon = 1.0` |
| VQ (commitment + codebook) | Frame 0 `_hyperbolic_vq` | Codebook + encoder outputs | `commitment_beta = 0.25` |
| Routing entropy | Chart router | Router sharpness | `w_entropy = 0.3` |
| Chart usage | Batch statistics | Chart center separation | `w_diversity = 1.0` |
| Codebook spread | Codebook geometry | Code separation | `w_codebook_spread = 0.05` |
| $z_n$ regularization | Structure filter | Nuisance magnitude | varies |
| Transition CE | Dynamics model | Transition MLP + (indirectly) encoder | `w_dyn_transition = 0.5` |
| Zeno smoothness | Consecutive transition predictions | Transition model + (indirectly) encoder | `w_zeno = 0.1` |
:::

### Adaptive Multipliers

:::{div} feynman-prose
Training a system with this many loss terms is an art. If the chart usage loss is too strong, all
charts get equal traffic but none of them specialize. If the VQ commitment is too strong, the
encoder freezes in place. If the dynamics weight is too strong, the codebook optimizes for
prediction at the expense of reconstruction quality.

The training loop uses adaptive multipliers that monitor online diagnostics and adjust loss weights
in response. For instance, if chart routing confidence (top-1 probability) drops below a target,
the confidence calibration multiplier ramps up. If code usage entropy falls below a threshold, the
code collapse penalty increases. These are documented in the Phase 1 adaptive state
(`init_phase1_adaptive_state`, `update_phase1_adaptive_state`).

The key principle: multipliers respond to *symptoms*, not to *epochs*. A time-based schedule
("increase dynamics weight at epoch 20") is fragile because it depends on learning rate, batch
size, and data distribution. A symptom-based schedule ("increase dynamics weight when reconstruction
has stabilized") adapts to whatever actually happens during training.
:::

The optimizer also uses differentiated learning rates for different parameter groups:

- **Chart centers:** `lr * lr_chart_centers_scale` (typically 0.1x base LR) --- chart centers
  should move slowly.
- **Codebook entries:** `lr * lr_codebook_scale` (typically 0.5x base LR) --- codes should track
  the encoder but not oscillate.
- **Dynamics transition model:** `lr_dyn_transition` (typically 3e-3) --- separate LR since this
  module has its own convergence dynamics.


(sec-diagnostics-encoder)=
## Diagnostics --- Knowing Your Encoder Is Working

:::{div} feynman-prose
Training neural networks is like flying an airplane at night. You cannot see where you are going.
You rely on instruments. If you do not have the right instruments, or you do not know how to read
them, you will crash.

Here are the instruments that matter for the shared-dynamics encoder.
:::

### Symbol Usage Statistics

The `_symbol_usage_stats` function computes joint $(K_{\mathrm{chart}}, K_{\mathrm{code}})$
statistics over a batch:

- **Active symbols:** How many of the $N_c \times K_{\mathrm{codes}}$ joint states are actually
  used? If only 10 out of 1024 states are active, the codebook has collapsed.

- **Symbol entropy:** $H = -\sum_s p(s) \log p(s)$. Maximum entropy means uniform usage across
  all states. Low entropy means a few states dominate. Track this and compare to
  $\log(N_c \cdot K_{\mathrm{codes}})$.

- **Symbol perplexity:** $\exp(H)$. The effective number of states. Easier to interpret than
  entropy.

- **Per-chart code entropy/perplexity:** The same statistics computed within each chart. Even if
  global entropy is high, individual charts might have collapsed codes.

```python
def _symbol_usage_stats(K_chart, K_code_dyn, num_charts, codes_per_chart):
    flat_state = K_chart.long() * codes_per_chart + K_code_dyn.long()
    counts = torch.bincount(flat_state.reshape(-1), minlength=num_states)
    probs = counts / counts.sum().clamp(min=1.0)
    entropy = -(probs * probs.clamp(min=1e-8).log()).sum().item()
    ...
```

### Closure Ratio Monitoring

The closure ratio ({prf:ref}`def-closure-ratio`) measures how well the transition model predicts
the next macro state. In the training loop, the key diagnostics are:

- **`dyn_trans_ce`:** Cross-entropy of the transition predictions. Should decrease over training.
- **`dyn_trans_acc`:** Fraction of correct next-state predictions. Should increase.
- **`dyn_code_flip_rate`:** Fraction of timesteps where the dynamics code changes. If this is
  very high, the codes are unstable (Zeno-like behavior at the code level).
- **`dyn_state_flip_rate`:** Fraction of timesteps where the *predicted* next state differs
  between consecutive predictions. High values suggest the transition model is uncertain.

### What to Look For During Training

:::{admonition} Training Health Checklist
:class: feynman-added tip

1. **Early training (epochs 0--10):** Reconstruction loss should drop. Symbol entropy should be
   moderate (not collapsed, not uniform). Chart usage should be spreading across charts.

2. **Mid training (epochs 10--50):** Dynamics accuracy should start climbing above chance
   ($1/(N_c \cdot K_{\mathrm{codes}})$). Zeno loss should decrease. Per-chart code entropy should
   be healthy (not all codes active, not all collapsed).

3. **Late training (epochs 50+):** Reconstruction loss should plateau. Dynamics accuracy should
   be substantially above chance. Code flip rate should be moderate (not zero, not near 1.0).
   Symbol perplexity should reflect the true complexity of the environment.

4. **Warning signs:**
   - Symbol entropy near zero: codebook collapse. Increase code collapse penalty or reduce
     commitment beta.
   - Dynamics accuracy stuck at chance: the codebook is not capturing dynamically relevant
     structure. Increase `w_dyn_transition`.
   - Zeno loss increasing: chart routing is becoming unstable. Check chart center separation
     and router margin losses.
   - Per-chart code entropy near zero but global entropy high: codes within each chart have
     collapsed to a single winner. Increase `w_code_collapse`.
:::

### Test-Set Evaluation

The `_test_eval_dynamics` function runs the full pipeline on held-out data:

- Reconstruction loss on all frames (not just frame 0).
- Dynamics CE, accuracy, Zeno, and code flip rate on sequences.
- Symbol usage statistics on the test set.

Comparing train and test symbol statistics reveals whether the codebook has overfit to training
sequences or learned genuinely transferable abstractions.


(sec-putting-it-together)=
## Putting It All Together

:::{div} feynman-prose
Let me give you the bird's-eye view. Here is the whole system in one picture: where the theory
lives, where the code lives, and which loss enforces which theoretical requirement. If you
understand this table, you understand the shared-dynamics encoder.
:::

### Theory-to-Code Correspondence

:::{div} feynman-added
| Theory Concept | Reference | Code Location | Loss Signal |
|---------------|-----------|---------------|-------------|
| Three-channel decomposition | {prf:ref}`def-three-channel-latent` | `PrimitiveAttentiveAtlasEncoder.forward` | Reconstruction + VQ + structure filter |
| Causal enclosure | {prf:ref}`def-causal-enclosure` | `SharedDynAtlasEncoder.dynamics_vq` | Transition CE (indirect via shared codebook) |
| Closure defect | {prf:ref}`def-closure-defect` | `compute_dynamics_markov_loss` | `dyn_trans_ce` monitors defect magnitude |
| Closure condition | {prf:ref}`def-causal-enclosure-condition` | Structure filter + enclosure probe | Enclosure loss pushes texture to be dynamics-free |
| Chart routing | {ref}`sec-tier-the-attentive-atlas` | `CovariantChartRouter` | Routing entropy + margin + consistency |
| Covariant attention | {ref}`sec-covariant-cross-attention-architecture` | `CovariantChartRouter._gamma_term` | Christoffel corrections improve routing |
| Hyperbolic VQ | {ref}`sec-the-shutter-as-a-vq-vae` | `_hyperbolic_vq` | Commitment + codebook loss |
| Zeno smoothness | {ref}`sec-the-equations-of-motion-geodesic-jump-diffusion` | `compute_dynamics_markov_loss` (zeno block) | JSD between consecutive transition predictions |
| Information bound | {ref}`sec-causal-information-bound` | VQ bottleneck ($\log_2 K$ bits max) | Codebook size limits channel capacity |
| Belief dynamics | {ref}`sec-filtering-template-on-the-discrete-macro-register` | `DynamicsTransitionModel` | Transition CE trains the Markov belief update |
| WFR geometry | {ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces` | Poincare ball operations throughout | Mobius addition, hyperbolic distance, barycenters |
| Hierarchical latents | {prf:ref}`def-hierarchical-latent` | $(K_{\mathrm{chart}}, K_{\mathrm{code}})$ factorization | Chart usage + code usage (two-level entropy) |
| Holographic coefficient | {prf:ref}`def-holographic-coefficient` | Codebook spread + radial calibration losses | Codebook geometry regularizers |
:::

### Summary Diagram

```{mermaid}
flowchart TB
    subgraph Encoder["SharedDynAtlasEncoder"]
        X["Observation x"] --> FE["Feature extraction"]
        FE --> V["v (Poincare ball)"]
        V --> ROUTE["CovariantChartRouter"]
        ROUTE --> |"K_chart, router_weights"| LOCAL["Chart-local coords"]
        V --> LOCAL
        LOCAL --> VQ["Shared codebook VQ"]
        VQ --> |"K_code, z_q"| SPLIT["Three-channel split"]
        LOCAL --> SPLIT
        SPLIT --> ZN["z_n (nuisance)"]
        SPLIT --> ZTEX["z_tex (texture)"]
        VQ --> ZGEO["z_geo"]
        ZN --> ZGEO
    end

    subgraph Losses["Loss Signals"]
        ZGEO --> DEC["Decoder"] --> RECON["Recon MSE"]
        VQ --> VQLOSS["VQ loss (commitment + codebook)"]
        ROUTE --> RLOSS["Routing losses (entropy, margin)"]

        VQ --> |"dynamics_vq (same codebook, zero VQ loss)"| DYN["DynamicsTransitionModel"]
        DYN --> DYNCE["Transition CE + Zeno smoothness"]
    end

    RECON --> TOTAL["Total loss"]
    VQLOSS --> TOTAL
    RLOSS --> TOTAL
    DYNCE --> TOTAL
    TOTAL --> BACK["Backward + optimizer step"]
```

:::{div} feynman-prose
And there it is. One encoder. One codebook. Two jobs --- reconstruct and predict. The theory says
these jobs should be compatible, and the shared codebook forces the network to find symbols where
they actually are. Every loss term maps to a theoretical requirement. Every diagnostic tells you
whether that requirement is being met.

The beautiful part is what is *not* here: no phased pipeline, no codebook copying, no delicate
handoff between training stages. Just one system learning to perceive and predict at the same time,
the way any competent agent should.
:::
