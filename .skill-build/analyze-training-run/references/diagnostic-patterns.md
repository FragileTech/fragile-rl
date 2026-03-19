# Diagnostic Patterns For Training Logs

Use this file when the run has many component metrics and you need a disciplined interpretation.

## 1. Start / Best / Break / Final

Always identify four anchor points when possible:

- `start`: the first logged epoch
- `best`: the best eval phase, not the lowest train loss
- `break`: the first clear degradation or collapse point
- `final`: the last logged epoch

If there is no eval, say so and rely on train-side evidence with lower confidence.

## 2. Evidence That Progress Is Real

Treat progress as real only when at least one of these is true:

- eval improves, not just train
- component losses improve together instead of one term dominating the gain
- router grounding metrics improve with the loss
- code usage remains healthy while loss drops
- train/eval gap stays controlled near the best phase

## 3. False Progress Patterns

### Main loss falls but symbolic metrics stay flat

Interpretation:

- optimization is happening, but not on the part of the model you care about

Typical evidence:

- `loss/main` falls
- `I_XK` stays near zero
- chart/code activity stays collapsed
- enclosure / markov remain flat or chance-like

Typical fixes:

- increase grounding pressure
- change model selection metric
- rebalance component weights

### Confidence rises while diversity disappears

Interpretation:

- router collapse, not clean learning

Typical evidence:

- `top1_prob_mean` near 1
- `H_K` near 0
- `active_code_charts` near 1
- code activity arrays show only one live route

Typical fixes:

- slow adversarial pressure
- increase diversity or usage regularization
- lower learning rate on router/codebook/chart-center parameters

### Enclosure looks chance-like

Interpretation:

- structured state is not predictive enough, or the target space collapsed

Typical evidence:

- enclosure CE flat around a simple occupied-state baseline
- enclosure accuracy barely moves
- texture-bearing heads do not help much over baseline

Typical fixes:

- improve symbol grounding first
- reduce pressure that destroys useful action or obs structure
- evaluate effective occupied targets, not only nominal state count

### Markov transition improves only slightly

Interpretation:

- transition model may be fitting a diffuse teacher without learning hard symbolic dynamics

Typical evidence:

- transition CE improves only a little
- hard chart accuracy stays near chance
- code accuracy is unstable
- target entropy stays high

Typical fixes:

- sharpen symbols before relying on macro dynamics
- compare CE to target entropy
- inspect whether targets are soft or hard

## 4. Grounding Heuristics

Treat symbols as weakly grounded when most of the following hold:

- `I_XK` is near zero for many epochs
- `top1_prob_mean` is near a uniform baseline
- code activity per chart is mostly 1
- active chart count does not track actual input variation

Treat symbols as healthier when:

- `I_XK` rises materially
- occupancy is balanced but not input-independent
- multiple codes per chart are active
- eval metrics stay strong while diversity remains non-trivial

## 5. Generalization Heuristics

Healthy:

- best eval occurs near a train plateau
- gap is modest near the best phase

Unhealthy:

- train keeps improving while eval worsens
- best phase occurs early and never returns
- later checkpoints only look better on train

## 6. Operational Checks

Always check:

- was the run completed?
- does a final checkpoint exist?
- was the best eval epoch actually saved?
- do checkpoint metrics reflect the same epoch as the weights?

If save cadence and eval cadence differ, say explicitly that checkpoint metadata may lag the actual weights.

## 7. Fix Prioritization

Prioritize fixes in this order unless evidence suggests otherwise:

1. Remove collapse or grounding failures
2. Protect the best eval phase with better checkpointing / early stopping
3. Rebalance objectives so the important component can improve
4. Tune secondary quality metrics after the main failure modes are gone
