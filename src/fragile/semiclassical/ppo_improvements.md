# PPO improvements from Fragile theory

## Executive summary

Our theory is a strict generalization of standard RL. In that view, vanilla PPO is the degenerate limit where:

- the latent geometry is flattened,
- the trust region lives in parameter-space KL instead of belief/state space,
- the reward field is assumed conservative and scalar,
- belief filtering is collapsed into a feed-forward state encoder,
- the Sieve is mostly absent,
- exploration is handled by a fixed entropy bonus instead of a governed temperature.

That means we should not ask "how do we replace PPO?" first. The right question is: which parts of Fragile give the biggest gain if we retrofit them into the current PPO baseline with minimal code motion?

For `src/fragile/semiclassical/ppo_continuous_action.py`, the best immediate improvements are:

1. add Sieve-lite diagnostics and update gates,
2. separate actor and critic timescales,
3. replace fixed entropy with a controlled temperature / noise schedule,
4. use the existing stable layer primitives (`SpectralLinear`, `NormGatedGELU`, `IsotropicBlock`),
5. add a cheap state-space sensitivity proxy so PPO is not only trusting parameter-space KL.

Anything beyond that starts to become a migration toward `src/fragile/semiclassical/geometric_ppo_continuous_action.py`, which is the more natural host for the full theory.

## Theory -> PPO mapping

The relevant theory hooks are:

- `docs/source/1_agent/01_foundations/02_control_loop.md`
  - PPO/TRPO are treated as a degenerate trust region: parameter-space Fisher/KL instead of state-space metric and WFR trust region.
- `docs/source/1_agent/04_control/01_exploration.md`
  - entropy is a controlled temperature, not just a fixed bonus coefficient.
- `docs/source/1_agent/04_control/02_belief_dynamics.md`
  - stable control is predict -> update -> project, not just sample -> optimize.
- `docs/source/1_agent/04_control/03_coupling_window.md`
  - stable learning requires an information-stability window and measurable rates.
- `docs/source/1_agent/02_sieve/01_diagnostics.md`
  - diagnostics should gate updates, not only be logged after the fact.
- `docs/source/1_agent/05_geometry/01_metric_law.md` and `05_geometry/04_equations_motion.md`
  - the metric acts as mass/preconditioner; high-risk regions should slow updates automatically.
- `docs/source/1_agent/06_fields/02_reward_field.md`
  - the critic is the exact conservative part of the reward field; curvature and varentropy should influence control temperature and caution.

## What the current file is missing

`src/fragile/semiclassical/ppo_continuous_action.py` is currently a clean PPO baseline, but from the theory perspective it has five major gaps:

| Current code area | What it does now | Fragile-theory gap | Consequence |
| --- | --- | --- | --- |
| `Args` and global hyperparameters | single actor/critic LR, fixed `ent_coef`, optional `target_kl` | no governor, no controlled temperature, no timescale separation | PPO is tuned by static scalars |
| `build_mlp` and `Agent` | plain `Linear + Tanh`, global `actor_logstd` | no gauge/Lipschitz control, no typed latent split, no separation between semantic action and execution noise | fragile updates, weak exploration control |
| rollout loop | only stores flat observations/actions/logprobs/values | no belief state, no filter/projection, no grounding checks | no coupling-window monitoring |
| update loop | PPO clip + value loss + entropy loss | trust region only in parameter-space ratio; no state metric, no actor gating, no diagnostics-driven scheduler | unstable updates are only weakly damped |
| logging | reward, KL, clipfrac, explained variance | no Sieve telemetry | no theory-aligned observability |

## Prioritized improvements to PPO in general

These are the main ways Fragile theory can make PPO better, ordered by how fundamental they are.

| Improvement family | Theory meaning | Rough cost | Main tradeoff |
| --- | --- | --- | --- |
| Sieve-guided PPO | hard or semi-hard update gates from diagnostics | Low | more training logic, less "pure" PPO |
| Temperature-governed PPO | entropy/noise becomes a controlled quantity | Low | more controller tuning |
| Two-timescale PPO | actor cannot outrun critic | Low | slower policy updates when critic is weak |
| Geometry-aware PPO | trust region uses state sensitivity, not only policy KL | Medium | extra backward passes or approximations |
| Conformal PPO | high-curvature value regions slow policy updates | Medium/High | noisy curvature estimates |
| Belief-filtered PPO | add predict/update/project structure | Medium/High | more architecture and state management |
| Reward-field PPO | separate exact value from curl/residual structure | High | more heads and more diagnostics |
| WFR PPO | trust region in transport/reaction space | Very high | substantially more math and compute |
| Atlas / macro-state PPO | typed `(K, z_n, z_tex)` latent control | Very high | this stops being "minimal PPO" |

## Most immediate improvements to `ppo_continuous_action.py`

### 1. Split actor and critic optimization, then gate actor updates

Current issue:

- one optimizer updates everything at once,
- actor and critic always train on the same schedule,
- the only guardrail is `target_kl`.

Immediate change:

- add `actor_learning_rate` and `critic_learning_rate`,
- use separate optimizers,
- optionally use different epoch counts for actor and critic,
- gate actor updates when:
  - `approx_kl` is too high,
  - explained variance is too low,
  - advantage SNR is too low,
  - clipfrac is persistently saturated.

Why this fits the theory:

- it is the cheapest implementation of the timescale hierarchy from the Sieve,
- it directly operationalizes "actor should not outrun critic."

Code touchpoints:

- `Args` in `ppo_continuous_action.py`
- optimizer creation
- minibatch update loop

Computational cost:

- Low
- almost no extra memory
- negligible extra wall-clock

Tradeoffs:

- training becomes more conditional and slightly less reproducible if gates are too aggressive,
- bad thresholds can freeze the actor too often.

### 2. Replace fixed entropy with a controlled temperature and a better noise model

Current issue:

- `ent_coef` is fixed,
- `actor_logstd` is a single global parameter, so exploration is not state-aware and not governed.

Immediate change:

- add `target_entropy`,
- update `ent_coef` online with a PI or SAC-style controller,
- replace the global `actor_logstd` with either:
  - a small state-dependent std head, or
  - a detached execution-noise controller that modulates noise around `actor_mean`.

Why this fits the theory:

- exploration is a temperature-governed control variable,
- policy semantics and execution noise should be less entangled,
- varentropy spikes can be used to slow cooling instead of collapsing exploration too early.

Code touchpoints:

- `Agent.actor_logstd`
- loss assembly
- logging

Computational cost:

- Low to Medium
- state-dependent std is a tiny extra head

Tradeoffs:

- target entropy becomes another controlled quantity to calibrate,
- if the controller is too aggressive, entropy will oscillate.

### 3. Add Sieve-lite diagnostics and use them to skip or scale minibatch updates

Current issue:

- the file logs only reward, KL, clipfrac, entropy, and explained variance,
- no diagnostic actually changes the update beyond `target_kl`.

Immediate change:

- log and gate with cheap online proxies:
  - ZenoCheck: minibatch KL and actor-mean drift,
  - ErgoCheck: entropy band error,
  - SNRCheck: advantage SNR and gradient SNR proxy,
  - QSLCheck: action-mean step norm or observation-space step proxy,
  - CostBoundCheck: large value magnitude or exploding returns,
  - action saturation: fraction of actions hitting env bounds.
- if a diagnostic fails:
  - skip actor step,
  - reduce LR,
  - shrink clip coefficient,
  - or temporarily raise temperature.

Why this fits the theory:

- the Sieve is supposed to be operational, not philosophical,
- PPO becomes much easier to debug when failures are typed instead of just seen as "bad reward."

Code touchpoints:

- rollout logging
- minibatch update loop
- TensorBoard outputs

Computational cost:

- Low
- these are mostly scalar summaries already available from the rollout/update tensors

Tradeoffs:

- more knobs,
- easier to overengineer if every metric becomes a gate.

### 4. Replace plain MLP blocks with the stable primitives already in the repo

Current issue:

- `build_mlp` uses unrestricted `nn.Linear + Tanh`,
- this gives no explicit Lipschitz control and does not use the existing geometric micro-architecture.

Immediate change:

- add an optional flag such as `--use_spectral_backbone`,
- replace `nn.Linear` with `SpectralLinear`,
- replace `Tanh` with `NormGatedGELU`,
- or use a small `IsotropicBlock` stack.

Why this fits the theory:

- the docs argue for gauge-covariant / Lipschitz-bounded primitives,
- this is a cheap way to bring the baseline closer to that discipline without rewriting it around a topoencoder.

Code touchpoints:

- `build_mlp`
- imports from `fragile.layers`

Computational cost:

- Low to Medium
- power iteration adds some overhead but not much for small PPO backbones

Tradeoffs:

- slightly slower forward pass,
- possibly lower raw expressivity at equal width,
- but usually better stability.

### 5. Add a cheap state-space sensitivity proxy and use it as a second trust signal

Current issue:

- PPO currently trusts only `ratio` / KL,
- this is exactly the parameter-space-only trust region that the theory says is incomplete.

Immediate change:

- compute a diagonal state sensitivity proxy on minibatches using cheap surrogates:
  - policy Fisher on observations: `E[(d/dx log pi)^2]`,
  - critic gradient RMS: `sqrt(E[(dV/dx)^2])`.
- combine them into a scalar trust score,
- then use that score to:
  - scale actor LR,
  - scale `clip_coef`,
  - or weight the actor loss down in high-sensitivity regions.

Why this fits the theory:

- it is the smallest practical step toward a state-space metric `G(z)`,
- it keeps PPO from treating all states as equally safe to perturb.

Code touchpoints:

- minibatch update loop
- optional extra autograd on `b_obs[mb_inds]`

Computational cost:

- Medium
- one or two additional backward-style gradient computations on minibatches

Tradeoffs:

- noisy estimates,
- more autograd complexity,
- but much closer to the intended theory than plain PPO clipping.

### 6. Add conformal caution from critic curvature

Current issue:

- the critic affects the actor only through advantages,
- not through local caution / inertia.

Immediate change:

- approximate curvature with:
  - Hessian diagonal if affordable, or
  - gradient RMS as a cheaper proxy,
- inflate caution in sharp-value regions by shrinking:
  - actor LR,
  - clip range,
  - or exploration std.

Why this fits the theory:

- high-curvature value regions should behave like high-mass regions,
- the agent should slow down near critical decision boundaries.

Code touchpoints:

- critic forward pass
- update controller

Computational cost:

- Medium to High
- true Hessian terms are expensive; diagonal surrogates are acceptable

Tradeoffs:

- curvature estimates can be noisy,
- excessive caution may slow learning in tasks with naturally sharp values.

## Recommended implementation order for this file

If the goal is "best gain per line changed", the order should be:

1. split actor/critic optimizers and add actor gating,
2. add entropy/noise controller,
3. add Sieve-lite diagnostics and logging,
4. swap in spectral/stable primitives,
5. add state-sensitivity trust proxy,
6. add conformal caution.

That sequence preserves the current PPO structure while making it more theory-aware.

## Improvements that are valid, but no longer "immediate"

These are real upgrades from the theory, but once we do them the baseline file is no longer the right abstraction.

### 7. Add a belief state and predict -> update -> project loop

What it means:

- stop treating the observation tensor as the full control state,
- maintain a filtered latent belief,
- project away unsafe or inconsistent mass before acting.

Why it matters:

- this is the real fix for partial observability and coupling-window control.

Cost:

- High

Tradeoff:

- much better semantics, much more code.

### 8. Add a typed latent state `(K, z_n, z_tex)` with a topoencoder

What it means:

- move from flat observation PPO to macro-symbolic control,
- let the policy act on macro state plus nuisance, not raw flat observations.

Why it matters:

- this is the clean route to grounded diagnostics like `I(X;K)` and `H(K)`.

Cost:

- Very high

Tradeoff:

- much more faithful to the theory, but this becomes a geometric PPO implementation, not a baseline PPO file.

### 9. Replace PPO's clip trust region with a WFR trust proxy

What it means:

- constrain updates in belief transport/reaction space rather than only policy ratio space.

Why it matters:

- this is the theory-correct generalization of PPO's trust region.

Cost:

- Very high

Tradeoff:

- conceptually right, implementation-heavy.

### 10. Decompose reward into conservative value plus curl/residual field

What it means:

- critic estimates exact/conservative structure,
- auxiliary head estimates non-conservative residual / cycle structure.

Why it matters:

- standard PPO silently assumes conservative reward structure,
- the theory says that is not always true.

Cost:

- High

Tradeoff:

- much richer control signals,
- but significantly more modeling and diagnostics.

## The practical boundary: when to stop editing this file

There is an important engineering boundary here:

- If we only want a stronger PPO baseline, keep editing `ppo_continuous_action.py`.
- If we want typed latents, belief geometry, atlas structure, geometric action decoding, or richer trust regions, the right path is to move that work into `geometric_ppo_continuous_action.py` rather than bloating the baseline.

So the immediate policy should be:

- retrofit governance and stable optimization into the baseline,
- do not try to shoehorn the full Fragile ontology into a flat PPO script.

## Concrete recommendation

For the next coding pass on `src/fragile/semiclassical/ppo_continuous_action.py`, I would do exactly this:

1. add separate actor/critic optimizers and actor-update gates,
2. add automatic entropy tuning and replace the global logstd with a better exploration controller,
3. add Sieve-lite metrics and TensorBoard logging,
4. add an optional `SpectralLinear` backbone,
5. add a cheap diagonal state-sensitivity trust proxy.

That gives the highest immediate leverage while staying within the current file's design. After that, the next real jump should be migration pressure toward `src/fragile/semiclassical/geometric_ppo_continuous_action.py`, not continued growth of the flat PPO baseline.
