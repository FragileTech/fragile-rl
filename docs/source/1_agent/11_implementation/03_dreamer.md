(sec-geometric-dreamer)=
# Geometric Dreamer --- Model-Based RL on the Poincare Ball

## TLDR

- `src/fragile/learning/rl/train_dreamer.py` trains a two-manifold Dreamer with one observation topoencoder and one action topoencoder.
- `GeometricActor` maps the observation-side symbolic state `(K_{\mathrm{chart}}, K_{\mathrm{code}}, z_n)` to an action-side structured state and reconstructs the canonical motor latent `action_z_geo`.
- `action_model.decoder(...)` turns that canonical action latent into a deterministic `action_mean`; rollout-time exploration adds scheduleable `sigma_motor` noise around that mean.
- Replay stores both executed `actions` and deterministic `action_means`; the trainer uses `action_means` by default, together with action-manifold traces such as `action_latents`, `action_router_weights`, `action_charts`, `action_codes`, and `action_code_latents`.
- The exact critic used by the trainer is `world_model.potential_net`, not the standalone `GeometricCritic` class.
- The trainer also learns a separate symbolic `MacroValueModel` over observation-state and action-state symbols. That macro backbone is part of the current implementation and participates in critic shaping, actor gating, and actor objectives.
- `RewardHead` models a conservative source density and a residual reward one-form, projects the residual off the exact critic covector, and contracts the result against the canonical action latent.
- Residual reward, imagined actor return, and exploration are all gated by closure, exact-field stiffness, exact-vs-direct force consistency, and macro-control calibration.
- Closure and no-leak diagnostics come from the observation world model plus `EnclosureProbe`; the action topoencoder is an action codec and supervision anchor, not a second transition model.
- `boundary.py` still contains auxiliary control-field modules, but the main Dreamer path documented here is the canonical-action-latent path in `train_dreamer.py`.

## Roadmap

1. Why geometric model-based RL?
2. Architecture overview: observation manifold, action manifold, value structure, and imagination.
3. Reward modeling on joint state-action latent geometry.
4. The canonical-action controller.
5. The value structure: shared exact critic and symbolic macro backbone.
6. Imagination training with trust-gated actor updates.
7. DM Control and Fractal Gas integration.
8. The training pipeline.
9. Diagnostics that matter in practice.
10. Theory-to-code correspondence.


(sec-why-geometric-mbrl)=
## 1. Why Geometric Model-Based RL

:::{div} feynman-prose
The current code is geometric in a stronger sense than simply putting a curved metric around an otherwise standard Dreamer.

There are two learned manifolds. The observation topoencoder turns environment observations into symbolic and geometric state on the observation manifold. The action topoencoder turns actuator commands into symbolic and geometric state on the action manifold. The actor does not emit a raw Euclidean action first and then decorate it afterward. It predicts action-side symbolic state and reconstructs an action-side latent point. That latent point is the motor variable the rest of the RL stack reads.

That design choice matters because it keeps one object fixed all the way through the implementation. The world model is conditioned on the canonical action latent. The reward head contracts its residual one-form against the canonical action latent. Replay stores the canonical action latent. Imagination rolls forward under the canonical action latent. The environment action is only the decoded realization of that motor-side latent state.

The other important current feature is that the code now has two value-facing structures, not one. The exact value field lives inside `world_model.potential_net` and anchors conservative reward, covectors, and screened-Poisson training. On top of that, the trainer also learns a symbolic macro control backbone over discrete observation/action symbols. So the current Dreamer is not only geometric; it is geometric plus an explicitly trained symbolic control layer.
:::

(rb-current-dreamer-status)=
:::{admonition} Dreamer Design
:class: info
`src/fragile/learning/rl/train_dreamer.py` currently follows this design:

- observations and actions each have their own `SharedDynTopoEncoder` manifold;
- `GeometricActor` predicts the action-side structured state from the observation-side structured state;
- the deployed motor variable is the canonical action latent `action_z_geo`;
- `action_model.decoder(...)` produces deterministic `action_mean`, while rollout-time exploration is added separately through `sigma_motor`;
- replay stores both executed actions and action-manifold traces, but training uses `action_means` by default;
- the exact critic is `world_model.potential_net`;
- a separate `MacroValueModel` learns a symbolic state-action value/reward backbone over macro states and action symbols;
- `RewardHead` models the residual reward sector after projecting away the exact critic covector;
- closure and no-leak diagnostics are driven by the observation world model plus `EnclosureProbe`, not by a separate action transition model;
- actor RL is trust-gated by imagination quality, exact-field certification, macro-control certification, scale trust, and stiffness trust.

Auxiliary boundary-control modules remain available in `boundary.py`, but they are not the main `train_dreamer.py` control path.
:::


(sec-dreamer-architecture-overview)=
## 2. Architecture Overview

The current perception-imagination-action loop has five coupled pieces:

```{mermaid}
flowchart LR
    subgraph Perception["Observation Manifold"]
        OBS["obs [B, D_obs]"] --> OENC["SharedDynTopoEncoder"]
        OENC --> OZ["z_geo, rw"]
        OENC --> OK["K_chart, K_code, z_n"]
    end

    subgraph Policy["Action Manifold"]
        OK --> ACTOR["GeometricActor"]
        ACTOR --> AZ["action_z_geo"]
        ACTOR --> AK["action chart/code, action_z_n, action_z_q"]
        AZ --> ADEC["action_model.decoder"]
        ADEC --> AMEAN["action_mean"]
    end

    subgraph Dynamics["Latent Dynamics"]
        OZ --> WM["GeometricWorldModel"]
        AZ --> WM
        WM --> OZN["z_next, rw_next, chart_logits, phi_eff"]
    end

    subgraph Value["Reward And Value"]
        OZ --> CRIT["critic = world_model.potential_net"]
        AZ --> RH["RewardHead"]
        OZ --> RH
        OZ --> MACRO["MacroValueModel"]
        AK --> MACRO
        OZN --> MACRO
    end

    subgraph Closure["Closure And No-Leak"]
        WM --> CHART["chart prediction"]
        CHART --> ENCLOSURE["EnclosureProbe"]
    end

    AMEAN --> ENV["dm_control or Fractal Gas collection"]
    ENV --> OBS
```

### Module Summary

:::{div} feynman-added
| Module | Location | Input | Output | Role in the current RL path |
|--------|----------|-------|--------|-----------------------------|
| `SharedDynTopoEncoder` | `src/fragile/learning/vla/shared_dyn/encoder.py` | observations | `z_geo`, `rw`, `K_chart`, `K_code`, `z_n`, `z_tex` | observation manifold and symbolic latent state |
| `SharedDynTopoEncoder` | `src/fragile/learning/vla/shared_dyn/encoder.py` | actions or `action_mean` replay traces | `action_z_geo`, `action_rw`, `action_K`, `action_K_code`, `action_z_n`, `action_z_q` | action manifold and replay supervision anchor |
| `GeometricActor` | `src/fragile/learning/rl/actor.py` | observation-side `(K_chart, K_code, z_n)` | action-side chart/code logits, `action_z_n`, `action_z_q`, `action_z_geo` | structured action policy |
| `compose_structured_state_with_atlas`, `symbolize_latent_with_atlas` | `src/fragile/learning/rl/action_manifold.py` | structured or geometric latents | symbolic/geometric conversions | bridge between atlas structure and latent points |
| `GeometricWorldModel` | `src/fragile/learning/vla/covariant_world_model.py` | `z_0`, canonical action latents, `rw_0` | latent rollout, router weights, chart logits, momenta, `phi_eff` | geometric imagination and transition prediction |
| `RewardHead` | `src/fragile/learning/rl/reward_head.py` | state latent, action latent, action code latent, router weights, exact covector | residual reward one-form, conservative source density, curl diagnostics | reward decomposition |
| `MacroValueModel` | `src/fragile/learning/rl/train_dreamer.py` | soft observation-state and action-state symbol distributions | symbolic reward/value tables and macro covectors | symbolic control backbone |
| `SequenceReplayBuffer` | `src/fragile/learning/rl/replay_buffer.py` | full episodes | contiguous training subsequences | replay over observations, action means, and latent action traces |
| `EnclosureProbe` | `src/fragile/learning/vla/losses.py` | chart embedding, action canonical, texture latent | closure and no-leak diagnostics | texture leakage audit |
| `GeometricActionEncoder`, `GeometricActionBoundaryDecoder` | `src/fragile/learning/rl/boundary.py` | explicit control-field variables | latent control-field objects and decoded actions | auxiliary experiments, not the main trainer |
| `GeometricCritic` | `src/fragile/learning/rl/critic.py` | state latent and router weights | scalar value | available in the package, but not instantiated by `train_dreamer.py` |
:::

### Typed Latents in the RL Path

The encoder still organizes latent state into the three-channel structure introduced earlier in the book:

$$
Z_t = (K_t, z_{n,t}, z_{\mathrm{tex},t}).
$$

The current RL path keeps that organization explicit on both manifolds:

- on the observation side, the actor consumes `chart_idx`, `code_idx`, and `z_n`;
- on the action side, the actor predicts action chart logits, action code logits, `action_z_n`, and reconstructs `action_z_geo` from the action atlas;
- replay stores action-side chart/code identities, action router weights, and action code latents in addition to the decoded action mean;
- texture stays outside the decision interface.

The canonical motor variable is therefore

$$
a_t^\star := z^{\mathrm{act}}_{\mathrm{geo},t},
$$

and that is the object the current world model, reward head, replay traces, and imagination logic all use.


(sec-reward-head)=
## 3. The Reward Head --- Predicting Rewards in Curved Space

:::{div} feynman-prose
The reward code now makes one important distinction very explicit.

The conservative part of reward is anchored by the exact value field. That part is not learned by giving a free residual network permission to fit everything. Instead, the trainer computes a value-backed conservative increment and asks the residual head to explain only what is left over after that conservative piece has been taken out.

So the reward head is not the whole reward model by itself. It is the non-exact sector together with the conservative source density used in the screened-Poisson constraint. The current implementation tries hard to keep that residual sector small, orthogonal to the exact field, and inactive until the exact field is actually usable.
:::

The reward model is implemented as `RewardHead`, not as a separate actor or critic branch. It consumes

- the current state latent `z_t`,
- the current chart routing weights `rw_t`,
- the action-manifold latent `action_z_t`,
- the action-manifold routing weights `action_rw_t`,
- the action-manifold code latent `action_z_{q,t}`,
- the canonical action latent `a_t^\star`.

The conservative replay increment comes from the exact critic:

$$
\hat r_{\mathrm{cons},t}
=
V(z_t, rw_t)
-
\gamma c_t V(z_{t+1}, rw_{t+1}),
$$

where $c_t = 1 - \mathrm{done}_t$ is the continuation factor used by the trainer.

The residual sector is modeled as a one-form-like covector field on joint state-action latent geometry:

$$
\hat r_{\mathrm{noncons},t}
=
\langle A_{\mathrm{res}}(z_t, a_t^\star), a_t^\star \rangle.
$$

In code, `RewardHead.decompose(...)` returns

- `reward_density`, the state-only conservative source term used in the screened Poisson loss,
- `reward_form_cov_raw`, the raw residual covector,
- `reward_form_cov`, the residual covector after projecting away the exact critic direction,
- `reward_form_exact_component`, the projected-out leakage onto the exact field,
- `reward_nonconservative`, the contraction of the projected residual covector with the canonical action latent,
- `reward_curl`, the exterior derivative of the projected residual one-form when curl diagnostics are enabled.

The scalar reward prediction is therefore

$$
\hat r_t
=
\hat r_{\mathrm{cons},t}
+
\hat r_{\mathrm{noncons},t}.
$$

The current trainer uses four kinds of pressure on that split:

$$
\mathcal{L}_{\mathrm{reward}}
=
\operatorname{MSE}(\hat r_t, r_t),
$$

$$
\mathcal{L}_{\mathrm{noncons}}
=
\operatorname{MSE}(\hat r_{\mathrm{noncons},t}, g_t(r_t - \hat r_{\mathrm{cons},t})),
$$

$$
\mathcal{L}_{\mathrm{cons\text{-}match}}
=
\operatorname{MSE}(\hat r_{\mathrm{cons},t}, r_t - g_t\hat r_{\mathrm{noncons},t}),
$$

plus exact-orthogonality, residual-norm, and residual-budget penalties.

The gate $g_t$ is not arbitrary. It is built from

- the norm of the exact covector field, which prevents the residual sector from opening before the conservative field is stiff enough,
- direct-vs-exact force agreement, which prevents the residual sector from compensating for a poorly calibrated conservative rollout field.

:::{admonition} Implementation Note: What The Residual Gate Is Doing
:class: feynman-added note
The residual gate in `_reward_nonconservative_gate(...)` is the current code's way of saying:

- do not trust the residual sector when the exact value field is nearly flat;
- do not trust the residual sector when the direct conservative rollout field does not yet agree with the exact scalar-gradient conservative field;
- only let the residual sector contribute once the conservative field is already behaving like a usable control object.
:::


(sec-geometric-actor)=
## 4. The Action Controller --- Observation Latent to Canonical Action Latent

:::{div} feynman-prose
The actor is deterministic in structure even when collection is exploratory in execution.

`GeometricActor` takes the observation-side symbolic state, predicts action-side chart and code probabilities, predicts an action-side nuisance coordinate, and reconstructs an action-side geometric latent from the bound action atlas. That reconstructed latent is the controller's actual motor output. The decoded actuator action is downstream of it.

This means exploration is not represented as the policy changing its latent semantics every time it samples. The policy emits a deterministic `action_mean` from `action_model.decoder(...)`, and rollout-time exploration is added afterward with the `sigma_motor` schedule. That keeps the action manifold itself as the semantic object and treats execution noise as a separate physical detail.
:::

The current actor path has three steps.

### 4.1 Structured Action Prediction

Given the observation-side symbolic state, the actor computes

$$
(K_t^{\mathrm{obs}}, K_{\mathrm{code},t}^{\mathrm{obs}}, z_{n,t}^{\mathrm{obs}})
\mapsto
(\ell_t^{\mathrm{chart}}, \ell_t^{\mathrm{code}}, z_{n,t}^{\mathrm{act}}),
$$

where `GeometricActor` produces chart logits, code logits, and action-side nuisance coordinates.

### 4.2 Canonical Action Latent

Those action-side symbolic predictions are then composed with the bound action atlas to reconstruct

$$
a_t^\star := z^{\mathrm{act}}_{\mathrm{geo},t}.
$$

The actor also exposes

- `action_chart_idx` and `action_code_idx`,
- `action_router_weights`,
- `action_z_q`, the action code latent,
- `action_z_n`, the action nuisance latent.

So the current controller is a structured symbol-to-symbol map with a geometric realization, not a flat action regressor.

### 4.3 Action Realization And Replay Supervision

The environment-facing action is produced by

$$
a_t = D_{\mathrm{act}}(a_t^\star, rw_t^{\mathrm{act}}),
$$

implemented by `action_model.decoder(...)`.

Two details in the current trainer matter:

- collection stores both the executed `actions` and the deterministic decoded `action_means`;
- replay training uses `action_means` by default, so the world model and action topoencoder are trained against the policy's deterministic motor realization rather than against rollout noise.

:::{admonition} Status Of The Policy Interface
:class: feynman-added note
`_policy_action(...)` in `train_dreamer.py` detaches the canonical action latent before decoding and returns detached rollout objects for collection and evaluation. During actor training, the code bypasses that helper and calls `GeometricActor` directly so gradients flow through imagined latent rollouts and the actor-side objectives.
:::


(sec-critic-as-potential)=
## 5. The Value Structure --- Exact Critic And Symbolic Macro Backbone

:::{div} feynman-prose
The current implementation has two value-facing objects, and it is important not to collapse them into one story.

The first is the exact critic: the shared value field inside `world_model.potential_net`. That field gives you conservative reward increments, exact covectors, screened-Poisson structure, stiffness certification, and exact-vs-direct force comparisons.

The second is the symbolic macro backbone: a separate `MacroValueModel` defined over discrete observation-state symbols and discrete action-state symbols. That macro model is not a duplicate critic. It is a transition-backed symbolic control scaffold built from the world model's latent rollout and the atlas's symbolic state distributions. In the current code, actor certification depends on both the exact field and this symbolic backbone.
:::

### 5.1 Exact Critic Shared With The World Model

In the main trainer,

$$
\texttt{critic} = \texttt{world\_model.potential\_net}.
$$

The conservative increment used throughout the trainer is

$$
\hat r_{\mathrm{cons},t}
=
V(z_t, rw_t) - \gamma c_t V(z_{t+1}, rw_{t+1}).
$$

The exact covector field is

$$
dV_t := \nabla_z V(z_t, rw_t),
$$

implemented by `_value_covector_from_critic(...)`.

The exact critic is trained with replay return anchoring and field-shaping losses. Schematically,

$$
\mathcal{L}_{\mathrm{critic}}
=
 w_{\mathrm{value}}\mathcal{L}_{\mathrm{value}}
+ w_{\mathrm{exact}}\mathcal{L}_{\mathrm{exact\text{-}increment}}
+ w_{\mathrm{Poisson}}(t)\mathcal{L}_{\mathrm{Poisson}}
+ w_{\mathrm{covector}}(t)\mathcal{L}_{\mathrm{covector\text{-}align}}
+ w_{\mathrm{stiff}}(t)\mathcal{L}_{\mathrm{stiffness}}
+ \text{on-policy and macro pullback terms}.
$$

The current code also distinguishes two conservative fields:

- a direct conservative field used by the fast rollout dynamics,
- an exact conservative field derived from the scalar value heads inside `potential_net`.

Force-consistency losses align the direct field to the exact field, and diagnostics log Hodge ratios for both.

### 5.2 The Symbolic Macro Backbone

`MacroValueModel` is currently defined inside `train_dreamer.py` as a symbolic state-action table with two learnable embeddings:

- `state_action_q`, the symbolic state-action value/control table,
- `state_action_reward`, the symbolic state-action reward table.

Observation-side macro states are the flattened symbolic indices

$$
s_t = (K^{\mathrm{obs}}_{\mathrm{chart},t}, K^{\mathrm{obs}}_{\mathrm{code},t}),
$$

and action-side macro actions are the flattened symbolic indices

$$
u_t = (K^{\mathrm{act}}_{\mathrm{chart},t}, K^{\mathrm{act}}_{\mathrm{code},t}).
$$

The current macro path does three things:

- it lifts latent states to soft symbolic state distributions with `_soft_symbolic_state_distribution(...)`;
- it builds a symbolic transition kernel from the world model rollout with `_macro_state_transition_distribution(...)`;
- it pulls symbolic value and control back into latent space by differentiating the lifted macro value/control with respect to the latent state.

So the macro backbone is not a separate world model. It is a symbolic control layer built on top of the existing observation atlas and world-model transition.

The macro losses now include

- macro value fitting,
- macro exact-increment fitting,
- macro pullback consistency,
- macro covector pullback consistency,
- macro transition cross-entropy,
- macro transition entropy sharpening,
- on-policy macro pullback and on-policy macro covector pullback when enabled.

The trainer gives `MacroValueModel` its own optimizer and scheduler. That matters because the symbolic backbone is not just a diagnostic table; it is explicitly trained as a separate subsystem.

:::{admonition} Implementation Note: `GeometricCritic`
:class: feynman-added note
`src/fragile/learning/rl/critic.py` still contains a standalone `GeometricCritic` module. The current `train_dreamer.py` path does not instantiate it. The exact critic used by training is the value field already living inside `world_model.potential_net`.
:::


(sec-imagination-training)=
## 6. Imagination Training --- Trust-Gated Canonical-Action Rollouts

:::{div} feynman-prose
Imagination is where the code combines geometry, symbolic structure, and control certification into one rollout.

At each imagined step, the code symbolizes the current observation-side latent state, predicts action-side symbolic state, reconstructs the canonical action latent, decodes an action mean for logging, rolls the world model forward under the canonical action latent, evaluates the exact conservative increment, evaluates the residual reward sector, and also evaluates the symbolic macro control backbone.

That gives the actor more than one signal, but it does not give it permission to trust every signal equally. The current trainer is explicitly conservative-first. Exact control has to certify itself. Macro control has to certify itself. Only then does the actor receive large return-driven updates. Curiosity is handled separately and is itself closure-gated.
:::

The imagination code is split between two functions:

- `_imagine(...)`, which produces detached diagnostic rollouts;
- `_imagine_actor_return(...)`, which differentiates the actor objective through latent rollouts.

The actor-facing rollout records

- latent states and router weights before and after each step,
- canonical action latents,
- decoded actions,
- conservative reward,
- residual reward,
- macro reward,
- reward-form covectors,
- exact critic covectors,
- macro covectors,
- world-model chart accuracy and router synchronization,
- curiosity statistics from chart varentropy.

### 6.1 The Imagined Objectives

The code accumulates four discounted quantities:

$$
J_{\mathrm{cons}} = \sum_{t=0}^{H-1} \gamma^t \hat r_{\mathrm{cons},t},
\qquad
J_{\mathrm{noncons}} = \sum_{t=0}^{H-1} \gamma^t \hat r_{\mathrm{noncons},t},
$$

$$
J_{\mathrm{macro}} = \sum_{t=0}^{H-1} \gamma^t \hat r_{\mathrm{macro},t},
\qquad
J_{\mathrm{cur}} = \sum_{t=0}^{H-1} \gamma^t \hat c_t,
$$

where `\hat c_t` is the chart-varentropy curiosity signal.

The conservative-first actor return used in the current implementation is

$$
J_{\mathrm{actor}}
=
J_{\mathrm{cons}}
+
\omega_{\mathrm{macro}} J_{\mathrm{macro}}
+
\tau_{\mathrm{cons}}^{\,p} J_{\mathrm{noncons}},
$$

where

- $\omega_{\mathrm{macro}}$ is the macro backbone weight derived from `actor_macro_backbone_weight` and the macro-control gate,
- $\tau_{\mathrm{cons}}$ is the conservative control gate,
- $p$ is `actor_return_nonconservative_power`.

### 6.2 Actor Gates In The Current Code

The current actor update uses several gates, not one.

First, imagination trust is computed from

- chart prediction accuracy,
- direct-vs-exact force error,
- policy/world-model router synchronization,
- exact conservative Hodge ratio.

Second, exact-control certification is computed from

- replay exact-increment calibration,
- on-policy covector-alignment calibration,
- on-policy exact-field stiffness calibration.

Third, macro-control certification is computed from

- macro exact-increment error,
- on-policy macro pullback error,
- on-policy macro signal scale.

Those pieces combine into the conservative control gate

$$
\tau_{\mathrm{cons}}
=
\tau_{\mathrm{return\text{-}trust}}
\cdot
\tau_{\mathrm{exact}}
\cdot
\tau_{\mathrm{macro}}.
$$

The return term then uses an additional scale and stiffness certificate:

$$
\tau_{\mathrm{return}}
=
\tau_{\mathrm{cons}}
\cdot
\tau_{\mathrm{scale}}
\cdot
\tau_{\mathrm{stiffness}}.
$$

So the current trainer does not let return-driven RL operate simply because imagination looks self-consistent. It also requires exact-field and macro-control calibration.

### 6.3 Natural Objective And Curiosity

The actor also optimizes a gauge-covariant natural objective built from the rollout velocity and a composite covector

$$
\Xi_t
=
 dV_t
 + \omega_{\mathrm{macro}}\, dQ^{\mathrm{macro}}_t
 - \tau_{\mathrm{cons}}^{\,p} A_{\mathrm{res},t}.
$$

That object is combined with the actor-side state metric proxy constructed from

- the exact covector scale,
- the actor's Fisher-like score scale over chart/code decisions,
- a small diagonal stabilizer.

Curiosity is handled separately. The code uses categorical chart varentropy from the world model's next-chart logits, then gates it by

- imagination trust,
- closure/no-leak quality from `EnclosureProbe`,
- actor scale trust,
- the complement of the conservative control gate.

So curiosity is not mixed into external reward. It is an additional policy objective that only becomes active when the policy is not yet strongly certified for conservative control and when the symbolic closure signals are good enough to interpret novelty structurally.

### 6.4 Replay Supervision And Old-Policy Trust

Replay supervision is still present, but it is now a bootstrap term rather than the permanent policy objective. The actor loss includes

- chart prediction cross-entropy against replay action symbols,
- code prediction cross-entropy against replay action symbols,
- action-side nuisance regression,
- a decaying replay-supervision scale,
- hyperbolic distance to `actor_old`,
- old-policy chart KL,
- old-policy code KL,
- scale barrier, world-model sync, and stiffness penalties,
- the imagined return term,
- the natural objective,
- the curiosity term.

After backpropagation, the trainer also applies a parameter-space trust-region rescaling step before the optimizer update. So the current actor path has both latent-geometry trust and parameter-space trust.


(sec-dmcontrol-integration)=
## 7. Integration With DM Control And Fractal Gas

:::{div} feynman-prose
The collection interface is still ordinary continuous control from the environment's point of view, but the trainer records much richer motor traces than a standard action-only replay buffer.

That is important because the action manifold is part of the implementation, not just a hidden helper. The replay buffer stores what actuator action was executed, what deterministic action mean the policy intended, what canonical action latent generated it, and what symbolic action-side state went with it. The world model, reward head, actor supervision, and macro backbone all reuse those traces.
:::

### Observation Path And Environment Setup

For dm_control tasks, `train_dreamer.py` does the following at runtime:

- loads the environment from `domain` and `task`,
- flattens the observation dictionary into one vector,
- infers `obs_dim` and `action_dim` from the environment and overrides the config if needed,
- optionally computes an `ObservationNormalizer` from seed episodes and uses it for later collection and training.

The trainer also supports environment-specific presets through `task_preset`. In the current code, `task_preset=auto` recognizes at least `cartpole-swingup` and `cartpole-balance`, and adjusts horizons, supervision strength, motor temperature schedule, and macro/critic weights accordingly.

### Rollout Modes

The current script supports three collection modes:

- random seed episodes,
- online dm_control rollouts, optionally in parallel through `VectorizedDMControlEnv`,
- optional `RoboticFractalGas` collection when `use_gas=True`.

During policy collection,

- the actor predicts `action_mean` from the canonical action latent,
- the trainer samples execution noise around that mean with `_sample_collection_action(...)`,
- the noise scale `sigma_motor` is scheduled by epoch and can also cool down as the exact-control gate improves.

Evaluation episodes are deterministic and use the decoded action directly.

### Replay Contents

The replay buffer stores full episodes and samples contiguous subsequences. The current episode format includes:

:::{div} feynman-added
| Key | Meaning |
|-----|---------|
| `obs` | flattened observations including the terminal observation |
| `actions` | executed environment actions |
| `action_means` | deterministic decoded policy actions before execution noise |
| `action_latents` | canonical action latents on the action manifold |
| `action_router_weights` | action-manifold routing weights |
| `action_charts` | action chart indices |
| `action_codes` | action code indices |
| `action_code_latents` | action-side code latents |
| `rewards` | environment rewards |
| `dones` | termination flags |
:::

A small but important implementation detail is that the trainer sets

$$
\texttt{action\_seq} = \texttt{batch.get("action\_means", batch["actions"])}.
$$

So replay supervision and world-model conditioning use the deterministic decoded action trace when it is available.


(sec-four-phase-pipeline)=
## 8. The Training Pipeline

### Overview

A single update is best read as a five-stage loop:

:::{div} feynman-added
| Substep | What updates | Main signals |
|---------|--------------|--------------|
| Observation/action encoding | both topoencoders, jump operators | Phase-1 reconstruction, atlas regularization, code usage, routing, nuisance regularization |
| Closure and no-leak | observation closure losses, `EnclosureProbe` | chart prediction, Zeno synchronization, enclosure defect penalties |
| Atlas synchronization | observation/action atlas consumers | chart centers and codebooks bound into world model, actor, reward head, and old actor |
| World model, exact critic, reward head, macro backbone | geometric dynamics, shared exact value field, `RewardHead`, `MacroValueModel` | geodesic loss, symbolic transition supervision, conservative/reward split, screened Poisson, exact/direct force consistency, macro transition and pullback losses |
| Actor | `GeometricActor` and `actor_old` snapshot | replay bootstrap, old-policy trust, natural objective, trust-gated imagined return, curiosity |
:::

### 8.1 Encoding, Closure, And No-Leak

The trainer always computes encoder-side losses for both manifolds unless the encoders are explicitly frozen. On top of the Phase-1 reconstruction and atlas losses, the current RL script adds observation-world-model closure through `_world_model_closure_losses(...)`.

That closure stage uses

- observation chart prediction from the world model,
- Zeno-style router smoothness,
- `EnclosureProbe` adversarial no-leak diagnostics using chart embeddings, canonical actions, and texture latents.

This is the current implementation of symbolic closure/no-leak in the RL path. There is no separate learned action transition model.

### 8.2 Atlas Synchronization

After the encoder and closure stage, `_sync_rl_atlas(...)` copies the current observation and action atlas objects into the RL consumers.

In particular, the current code binds

- observation chart centers into the world model and critic tokenizers,
- action chart centers and action codebook into `GeometricActor`,
- action chart centers into `RewardHead.action_chart_tok`,
- the same action atlas into `actor_old`.

That synchronization step is essential because the actor reconstructs `action_z_geo` from atlas buffers rather than by owning a separate action atlas of its own.

### 8.3 World Model And Reward Update

The world model is trained on detached encoder outputs. The replay action input used in the world-model stage is the action latent encoded from replay `action_means` by the action topoencoder.

The world-model stage includes

- geodesic latent prediction loss,
- chart prediction loss,
- symbolic state/code supervision for predicted next states,
- reward reconstruction,
- conservative-match pressure,
- residual reward fitting,
- residual exact-orthogonality,
- exact/direct force consistency,
- momentum, energy, and Hodge losses.

The reward split is explicitly conservative-first. The trainer first computes the exact conservative target, then gates the residual sector, and only then lets the residual explain replay reward left over after that exact piece.

### 8.4 Exact Critic And Macro Backbone Update

The exact critic and the macro backbone are separate subsystems in the current code.

The exact critic stage trains the shared value field with

- replay return targets,
- exact-increment targets,
- screened Poisson consistency,
- covector-alignment losses,
- stiffness losses,
- on-policy covector and stiffness losses,
- macro pullback terms that force the exact field to stay compatible with the symbolic backbone.

The macro backbone stage trains `MacroValueModel` with its own optimizer using

- macro value targets,
- macro exact-increment targets,
- macro pullback losses,
- macro covector pullback losses,
- transition cross-entropy,
- transition entropy sharpening,
- optional on-policy pullback and on-policy covector pullback losses.

So the current trainer really has both a shared exact critic and a separately optimized symbolic control backbone.

### 8.5 Actor Update

The actor stage still starts with replay supervision on action symbols and action-side nuisance coordinates, but that supervision is explicitly decayed by `_actor_supervise_scale(...)` once trust-gated RL opens.

The full actor loss currently combines

$$
\mathcal{L}_{\mathrm{actor}}
=
\mathcal{L}_{\mathrm{supervise}}
+
\mathcal{L}_{\mathrm{old\text{-}policy}}
+
\mathcal{L}_{\mathrm{scale}}
+
\mathcal{L}_{\mathrm{sync}}
+
\mathcal{L}_{\mathrm{stiffness}}
+
\mathcal{L}_{\mathrm{curiosity}}
+
\mathcal{L}_{\mathrm{natural}}
+
\mathcal{L}_{\mathrm{return}}.
$$

Operationally, the return term only runs periodically, only after warmup, and only when the combined trust and certification gates are nonzero.

The actor stage also updates `actor_old` after each optimizer step, so old-policy trust is measured against the most recent pre-update actor snapshot.


(sec-dreamer-diagnostics)=
## 9. Diagnostics --- Is Your Agent Learning?

:::{div} feynman-prose
The current trainer logs enough information that you can usually tell which subsystem is failing before reward collapses completely.

That is important here because the implementation is no longer one monolithic Dreamer loss. There are encoder and closure failures, conservative-field failures, macro-control failures, and actor-trust failures, and they do not all show up in the same metric.
:::

### Representation And Closure Health

- `chart/usage_entropy`, `chart/active_charts`, `chart/active_symbols`
- `chart/router_confidence`, `chart/top2_gap`
- `action_chart/usage_entropy`, `action_chart/active_symbols`
- `closure/L_total`, `closure/L_obs_state`, `closure/L_obs_zeno`
- `closure/enclosure_acc_full`, `closure/enclosure_acc_base`
- `closure/enclosure_defect_acc`, `closure/enclosure_defect_ce`

These tell you whether the observation and action atlases remain alive, whether symbolic closure is grounded, and whether texture leakage is being suppressed.

### World-Model And Reward Health

- `wm/L_geodesic`, `wm/L_chart`, `wm/L_code`, `wm/L_symbol`
- `wm/L_reward`, `wm/L_reward_nonconservative`, `wm/L_reward_conservative_match`
- `wm/L_reward_exact_orth`, `wm/L_reward_nonconservative_norm`, `wm/L_reward_nonconservative_budget`
- `wm/reward_nonconservative_gate`, `wm/reward_nonconservative_gate_stiffness`, `wm/reward_nonconservative_gate_force`
- `wm/force_exact_rel_mean`, `wm/force_task_exact_rel_mean`, `wm/force_risk_exact_rel_mean`
- `wm/reward_curl_norm_mean`, `wm/reward_form_metric_norm_mean`

These tell you whether the transition model stays grounded, whether the residual reward sector is under control, and whether the direct conservative field is matching the exact field closely enough.

### Exact Critic Health

- `critic/L_value`, `critic/L_poisson`, `critic/L_exact_increment`
- `critic/L_covector_align`, `critic/L_stiffness`
- `critic/exact_covector_norm_mean`, `critic/stiffness_target`, `critic/stiffness_certified`
- `critic/calibration_ratio`
- `critic/on_policy/L_covector_align`, `critic/on_policy/L_stiffness`
- `critic/on_policy/exact_covector_norm_mean`, `critic/on_policy/calibration_ratio`
- `geometric/hodge_conservative_direct`, `geometric/hodge_conservative_exact`

These tell you whether the exact value field is actually behaving like a usable conservative control object rather than a weak scalar regressor.

### Macro Backbone Health

- `macro/L_macro`, `macro/L_value`, `macro/L_transition`, `macro/L_transition_entropy`
- `macro/L_pullback`, `macro/L_covector_pullback`
- `macro/exact_increment_abs_err`, `macro/target_scale`
- `macro/transition_acc`, `macro/transition_ce`, `macro/transition_sharpen_gate`
- `macro/on_policy/L_pullback`, `macro/on_policy/L_covector_pullback`
- `macro/on_policy/value_std`, `macro/on_policy/exact_increment_abs_err`

These tell you whether the symbolic control scaffold is predictive, calibrated, and strong enough to be used as part of actor certification.

### Actor And Imagination Health

- `actor/L_supervise`, `actor/supervise_scale`
- `actor/L_old_policy_geodesic`, `actor/L_old_policy_chart_kl`, `actor/L_old_policy_code_kl`
- `actor/return_trust_used`, `actor/return_gate`, `actor/return_applied`
- `actor/exact_control_gate`, `actor/macro_control_gate`
- `actor/curiosity_gate`, `actor/curiosity_closure_gate`
- `actor/L_natural`, `actor/gauge_covector_norm_mean`, `actor/macro_covector_norm_mean`
- `actor/state_alpha`, `actor/state_beta_pi`, `actor/state_scale_trust`, `actor/stiffness_trust`
- `imagination/reward_mean`, `imagination/discounted_reward_mean`, `imagination/nonconservative_return_mean`
- `imagination/exact_boundary_mean`, `imagination/terminal_value_mean`, `imagination/full_return_mean`
- `imagination/policy_chart_acc`, `imagination/policy_router_sync_mean`, `imagination/policy_force_rel_err_mean`

These tell you whether the actor is still mostly in bootstrap mode, whether exact and macro control have certified RL updates, and whether imagined trajectories are actually useful.

### How To Read The Current Failure Modes

:::{div} feynman-added
| Failure pattern | First metrics to inspect |
|----------------|--------------------------|
| residual reward opens too early | `wm/reward_nonconservative_gate`, `wm/reward_exact_covector_norm_mean`, `wm/force_exact_rel_mean` |
| actor stays stuck in behavior cloning | `actor/supervise_scale`, `actor/return_trust_used`, `actor/exact_control_gate`, `actor/macro_control_gate` |
| symbolic backbone is not grounded | `closure/obs_state_acc`, `closure/enclosure_defect_acc`, `macro/transition_acc`, `macro/transition_sharpen_gate` |
| exact conservative field is too weak | `critic/exact_covector_norm_mean`, `critic/stiffness_target`, `critic/stiffness_certified` |
| imagination drifts away from the atlas | `imagination/policy_router_sync_mean`, `actor/policy_router_sync_mean`, `wm/chart_acc` |
:::


(sec-dreamer-theory-code)=
## 10. Theory-to-Code Correspondence

:::{div} feynman-prose
The table below states the current implementation in the same terms the code now uses.
:::

:::{div} feynman-added
| Theory | Reference | Current code | Status |
|--------|-----------|--------------|--------|
| Three-channel latent | {ref}`sec-the-shutter-as-a-vq-vae` | topoencoders build chart/code structure, nuisance, texture, and geometric latents | Implemented |
| Two-manifold RL state | {ref}`sec-the-shutter-as-a-vq-vae` | separate observation and action `SharedDynTopoEncoder` modules | Implemented |
| Canonical action latent | {ref}`sec-the-boundary-interface-symplectic-structure` | `a_t^\star = action_z_geo` is the motor variable used by replay, reward, and dynamics | Implemented |
| Action realization | {ref}`sec-the-boundary-interface-symplectic-structure` | `action_model.decoder(a_t^\star, rw_t^{act})` produces deterministic `action_mean` | Implemented |
| Replay-grounded motor traces | {ref}`sec-the-boundary-interface-symplectic-structure` | replay stores executed actions, deterministic `action_means`, canonical action latents, action symbols, and action code latents | Implemented |
| Observation-world-model closure | {prf:ref}`def-causal-enclosure` | closure and no-leak are enforced with world-model chart prediction plus `EnclosureProbe` | Implemented |
| Exact conservative value field | {ref}`sec-the-reward-field-value-forms-and-hodge-geometry` | `critic = world_model.potential_net` supplies `V`, `dV`, exact increments, and screened-Poisson structure | Implemented |
| Exact-vs-direct conservative certification | {ref}`sec-the-reward-field-value-forms-and-hodge-geometry` | force-consistency losses and direct/exact Hodge diagnostics compare rollout forces to the exact scalar-gradient field | Implemented |
| Residual reward one-form | {prf:ref}`def-reward-1-form` | `RewardHead` predicts residual covector, conservative source density, and curl diagnostics | Implemented |
| Conservative-first reward split | {prf:ref}`def-reward-1-form` | residual sector is projected off the exact covector and gated by stiffness and force agreement | Implemented |
| Symbolic macro control backbone | {prf:ref}`def-maxent-rl-objective-on-macrostates` | `MacroValueModel` lifts symbolic observation/action states into reward, value, and control pullback objectives | Implemented |
| Transition-backed macro control | {ref}`sec-wfr-boundary-conditions-waking-vs-dreaming` | macro transitions come from the world-model rollout plus soft symbolic state distributions | Implemented |
| Trust-gated actor return | {ref}`Link <rb-barriers-trust-regions>` | actor return requires imagination trust, exact-control certification, macro-control certification, scale trust, and stiffness trust | Implemented |
| Old-policy trust region | {ref}`Link <rb-barriers-trust-regions>` | hyperbolic old-policy anchor, chart/code KL penalties, and parameter-space trust-region rescaling | Implemented |
| Curiosity from predictive uncertainty | {prf:ref}`def-maxent-rl-objective-on-macrostates` | chart varentropy drives a separate curiosity term gated by closure and trust | Implemented |
| Auxiliary explicit-control boundary modules | {ref}`sec-the-boundary-interface-symplectic-structure` | `boundary.py` keeps explicit control-field modules for side experiments | Available |
| Standalone geometric critic module | {ref}`sec-the-reward-field-value-forms-and-hodge-geometry` | `GeometricCritic` exists in the package but is not used by `train_dreamer.py` | Available |
:::

### Practical Summary

This Dreamer implementation should currently be understood as:

1. A two-manifold geometric model-based RL system with separate observation and action atlases.
2. A canonical-action-latent controller whose environment action is a decoded realization of the action manifold state.
3. A conservative-first reward/value stack built from a shared exact critic plus a residual reward one-form.
4. A symbolic macro control backbone layered on top of the geometric world model.
5. A trust-gated actor update that only uses large RL gradients once exact and macro control are both sufficiently calibrated.

:::{div} feynman-prose
So the right mental model for the current code is not just "Dreamer on a curved latent space." It is a two-manifold geometric Dreamer with one exact value field, one residual reward sector, one symbolic macro control scaffold, and one actor whose return updates are certified before they are trusted. The observation manifold tells the policy where it is. The action manifold tells the policy what motor-side state it is choosing. The exact value field tells the trainer what conservative reward structure is already justified. The macro backbone tells the trainer whether symbolic control is calibrated enough to support the actor. The residual sector and curiosity path stay visible, but both are explicitly gated.

That is the implementation currently living in `src/fragile/learning/rl/`.
:::
