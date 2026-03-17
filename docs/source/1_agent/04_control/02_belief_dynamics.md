(sec-belief-dynamics-prediction-update-projection)=
# Belief Dynamics: Prediction, Update, Projection

## TLDR

- Belief dynamics is a **predict → update → project** loop: forecast with the world model, assimilate observations, then
  project away unsafe/infeasible mass via the Sieve.
- Treat projection as a first-class operation: it is what makes filtering compatible with hard constraints.
- Use small discrete examples (belief vectors/matrices) to sanity-check implementations before scaling up.
- Keep notation local: this chapter is a bridge between the MaxEnt objective and the coupling-window stability theorem.
- If you keep the GKSL/Lindblad analogy, treat it as intuition only and keep the mapping explicit.

## Roadmap

1. Prediction and Bayes update on the macro state.
2. Sieve projection / reweighting and what it guarantees.
3. Correspondence table linking filtering objects to control-loop components.

:::{div} feynman-prose
Let me tell you what this chapter is really about: how an agent changes its mind.

You might think changing your mind is straightforward. You see something new, you update your beliefs, done. But here's the thing---and this is what makes the problem interesting---when you're a bounded agent operating in the real world, you can't just casually update your beliefs. Updates have to respect two hard constraints:

1. **You only learn from what you actually see.** The boundary---your sensors, your observations---is your only window into reality. You can't update your beliefs based on information you don't have.

2. **Some beliefs are dangerous.** If your belief distribution starts putting weight on states that would lead to catastrophic actions, you need to project that mass away before it causes trouble.

This is the "filtering + projection" story: first you do the normal Bayesian thing (predict what's coming, then correct based on what you see), and then you do the safety thing (throw out the beliefs that violate your constraints). Both steps are *irreversible*---you can't undo an observation, and you can't un-project a constraint. That irreversibility is fundamental. It's what makes online learning different from just rolling a simulator forward.
:::

(rb-bayes-filter)=
:::{admonition} Researcher Bridge: Bayes Filter with Safety Projection
:class: info
The predict-update loop is standard HMM/POMDP filtering. The extra step is projection by the Sieve, which removes or downweights unsafe belief mass. Think "Bayes filter plus constraints."
:::

:::{div} feynman-prose
Sections 2-9 describe geometry, metrics, and effective macro dynamics. What they do *not* yet encode is the irreversibility of online learning: boundary observations and constraint enforcement are not invertible operations. This section states the belief-evolution template directly as **filtering + projection** on the discrete macro register.
:::

:::{div} feynman-prose
**Relation to prior work.** The predict-update recursion below is standard Bayesian filtering for discrete latent states (HMM/POMDP belief updates) {cite}`rabiner1989tutorial,kaelbling1998planning`. The additional ingredient emphasized here is the explicit **projection/reweighting layer** induced by safety and consistency checks ({ref}`sec-diagnostics-stability-checks`): belief updates are not just "Bayes + dynamics", but "Bayes + dynamics + constraints".
:::

(sec-why-purely-closed-simulators-are-insufficient)=
## Why Purely Closed Simulators Are Insufficient

:::{div} feynman-prose
Here's a thought experiment that I find clarifying. Imagine you built a perfect internal simulator---a complete model of the world that you can run forward in your head. You start it in some state, you simulate physics, you predict the future. Beautiful.

But wait. Where do the boundary observations come in?

See, your internal simulator is running in your head. The actual world is out there, doing its own thing. If your simulator and the world start off synchronized (which they won't be, but pretend), they'll drift apart. Small errors accumulate. Your simulator has no idea about the actual forces hitting the actual robot, the actual wind, the actual perturbations from other agents.

The only way to fix this is to *assimilate* boundary observations---to look at what you actually see and correct your internal simulation accordingly. And that correction is not invertible. Once you've observed $x_{t+1}$, your belief state has changed in a way that can't be undone from the post-update state alone. You've irreversibly collapsed your uncertainty.

The same thing happens with safety constraints. If you discover that certain belief states would lead to violating a cost bound, you have to project that probability mass away. That's another irreversible operation. You can't just "un-project" and get back to where you were.

This is why purely closed internal simulation isn't enough. An agent has to be *open* to its boundary, and that openness introduces irreversibility at every step.
:::

:::{admonition} The Two Irreversibilities
:class: feynman-added note
A purely closed internal simulator can roll forward hypotheses, but it cannot *incorporate new boundary information* without a non-invertible update. Two irreversibilities are unavoidable:
1. **Assimilation:** boundary observations $x_{t+1}$ update the macro belief (Bayesian correction).
2. **Constraint enforcement:** the Sieve applies online projections/reweightings that remove unsafe/inconsistent mass (Gate Nodes / Barriers).

Both operations are information projections: they reduce uncertainty and/or discard parts of state-space mass in a way that cannot be undone from the post-update state alone.
:::

(sec-filtering-template-on-the-discrete-macro-register)=
## Filtering Template on the Discrete Macro Register

:::{div} feynman-prose
Alright, let's get concrete. We have a discrete macro register---a finite set of symbols $\mathcal{K}$ that represent the "big picture" states. At any moment, the agent has a *belief* over which macro-symbol is active. This belief is just a probability distribution: how much weight do I put on each possible macro-state?

The notation $p_t \in \Delta^{|\mathcal{K}|-1}$ looks scary but it just means "$p_t$ is a probability distribution over $|\mathcal{K}|$ states"---the $\Delta$ is the probability simplex, and the superscript counts the degrees of freedom (one less than the number of states, since probabilities sum to one).

Now, here's the two-step dance that happens at every timestep:
:::

:::{div} feynman-prose
Let $p_t\in\Delta^{|\mathcal{K}|-1}$ be the macro belief over $K_t$.
:::

:::{admonition} Step 1: Prediction (The Model Step)
:class: feynman-added tip

**What it is:** Use your learned dynamics model to forecast where you think you'll be next.

Given the learned macro kernel $\bar{P}(k'\mid k,a_t)$ ({ref}`sec-conditional-independence-and-sufficiency`), define the one-step predicted belief

$$
\tilde p_{t+1}(k') := \sum_{k\in\mathcal{K}} p_t(k)\,\bar{P}(k'\mid k,a_t).

$$

**In plain words:** "For each macro-state $k'$ I might end up in, sum over all the ways I could get there. The ways are weighted by (a) how much I currently believe I'm in state $k$, and (b) how likely my model says the transition $k \to k'$ is given action $a_t$."

This is matrix-vector multiplication in disguise: $\tilde{p}_{t+1} = \bar{P}^T p_t$.
:::

:::{admonition} Step 2: Update (The Observation Step)
:class: feynman-added tip

**What it is:** Incorporate what you actually observed to correct your prediction.

Given an emission/likelihood model $L_{t+1}(k'):=p(x_{t+1}\mid k')$ (or any calibrated score proportional to likelihood), the posterior belief is

$$
p_{t+1}(k')
:=
\frac{L_{t+1}(k')\,\tilde p_{t+1}(k')}{\sum_{j\in\mathcal{K}} L_{t+1}(j)\,\tilde p_{t+1}(j)}.

$$

**In plain words:** "Take my predicted belief $\tilde{p}_{t+1}(k')$, weight it by how well state $k'$ explains what I actually observed, and normalize so it's still a probability distribution."

This is Bayes' rule. The denominator is just there to make things sum to one.
:::

:::{div} feynman-prose
This is the standard Bayesian filtering recursion for a discrete latent state (HMM/POMDP belief update) {cite}`rabiner1989tutorial,kaelbling1998planning`. Units: probabilities are dimensionless; log-likelihoods and entropies are measured in nats.

Now here's what I want you to notice: this two-step dance---predict, then update---is completely standard. Any textbook on hidden Markov models will show you this. What's *not* standard is what comes next: the projection step. That's where safety enters the picture.
:::

## Worked Example: A Tiny Belief Update + Sieve Projection

Take three macrostates $\mathcal{K}=\{k_1,k_2,k_3\}$.

1. **Start with a prior belief:**

   $$
   p_t = [0.6,\ 0.3,\ 0.1].
   $$

2. **After the model prediction step** (apply $\bar{P}^T$), suppose you get:

   $$
   \tilde p_{t+1} = [0.5,\ 0.4,\ 0.1].
   $$

3. **Assimilate an observation** with likelihoods:

   $$
   L_{t+1} = [0.1,\ 0.8,\ 0.1].
   $$

   Elementwise multiply and normalize:

   $$
   p_{t+1} \propto L_{t+1}\odot \tilde p_{t+1} = [0.05,\ 0.32,\ 0.01],
   $$
   so $\sum_j = 0.38$ and

   $$
   p_{t+1} \approx [0.132,\ 0.842,\ 0.026].
   $$

4. **Apply a Sieve projection.** If a barrier marks $k_3$ infeasible (e.g., cost bound exceeded), hard projection does:

   $$
   p'_{t+1}(k)\propto p_{t+1}(k)\cdot \mathbb{I}[\text{feasible}(k)],
   $$

   which yields

   $$
   p'_{t+1} \approx [0.135,\ 0.865,\ 0].
   $$

This is the essential pattern: Bayes updates move mass toward what explains observations; the Sieve then removes mass that
would be unsafe or inconsistent with runtime contracts.

(sec-sieve-events-as-projections-reweightings)=
## Sieve Events as Projections / Reweightings

:::{div} feynman-prose
The Sieve is our safety mechanism---a collection of checks that monitor whether the agent's beliefs and actions are staying within acceptable bounds. When a check fails, we don't just log it and move on. We *modify the belief state* to push probability away from the dangerous regions.

There are two flavors of this, and they're worth understanding separately.
:::

:::{admonition} Hard Projection: The Binary Firewall
:class: feynman-added example

**Hard projection (mask + renormalize):**

$$
p'_{t}(k)\propto p_t(k)\cdot \mathbb{I}\!\left[\text{feasible}(k)\right].

$$

**What this means:** Some states are simply forbidden. Maybe state $k$ would violate a cost budget ($V(k) > V_{\max}$). Maybe it would put you in an irrecoverable situation. Whatever the reason, we set the belief mass on that state to exactly zero and renormalize what's left.

**The picture:** Imagine your belief is a distribution over a bunch of boxes. Hard projection says "these boxes are off-limits" and sweeps all the probability out of them, redistributing it among the allowed boxes.

Example: feasibility defined by a cost budget $V(k)\le V_{\max}$ (CostBoundCheck).
:::

:::{admonition} Soft Reweighting: The Exponential Push
:class: feynman-added example

**Soft reweighting (exponential tilt):**

$$
p'_t(k)\propto p_t(k)\,\exp\!\left(-\lambda\cdot \text{penalty}(k)\right),

$$

which implements a differentiable "push away" from unstable regions.

**What this means:** Instead of a hard cutoff, we continuously downweight states based on how bad they are. High-penalty states get exponentially suppressed; low-penalty states remain mostly untouched.

**The picture:** Imagine the penalty as a "badness score." The exponential tilt says "I'll tolerate some badness, but exponentially less as things get worse." This is smoother than hard projection and plays nicely with gradient-based learning.
:::

:::{div} feynman-prose
These are classical constrained-inference moves (mirror descent / I-projection style), and they are the belief-space counterpart of the Gate Nodes. The key insight is that projection happens *in belief space*---we're not directly moving the agent, we're moving its *beliefs about where it is*. But since actions depend on beliefs, this indirectly shapes behavior.
:::

(sec-over-under-coupling-as-forgetting-vs-ungrounded-inference)=
## Over/Under Coupling as Forgetting vs Ungrounded Inference

:::{div} feynman-prose
Now we come to a beautiful tension at the heart of belief dynamics. Your agent needs to be coupled to its boundary---to the stream of observations coming in from the world. But how much coupling is the right amount?

Too little coupling, and your agent starts living in its own head. Its internal model rolls forward, making predictions about what it thinks will happen, but those predictions drift further and further from reality. This is **ungrounded inference**---Mode D.C in our diagnostic taxonomy---and it's a recipe for disaster.

Too much coupling, and your agent becomes reactive and forgetful. Every little observation overwhelms its beliefs, the macro register can't maintain stable structure, and the agent loses the ability to reason about the future. This is **symbol dispersion**---the agent's internal "currency" of macro-states stops meaning anything coherent.

The coupling window we'll discuss in Theorem {prf:ref}`thm-information-stability-window-operational` is the Goldilocks zone: enough coupling to stay grounded, not so much that you lose structure. The Sieve (Sections 3-6) is the control layer that keeps the agent inside this window.
:::

:::{admonition} The Coupling Dilemma
:class: feynman-added warning

The coupling window in Theorem {prf:ref}`thm-information-stability-window-operational` reflects a fundamental trade-off:
- **Over-coupling:** noisy or overly aggressive updates drive mixing; the macro register loses stable structure (forgetting / symbol dispersion).
- **Under-coupling:** insufficient boundary information causes internal rollouts to dominate (model drift / ungrounded inference; Mode D.C).

There's no free lunch here. You must balance grounding against stability.
:::

(sec-optional-operator-valued-belief-updates)=
## Optional: Operator-Valued Belief Updates (GKSL / "Lindblad" Form)

:::{div} feynman-prose
This section is optional, and I want to be upfront about why it's here. The mathematics of GKSL (Lindblad) evolution comes from quantum mechanics, where it describes how open quantum systems evolve when they interact with an environment. But you don't need to care about quantum physics to find this useful.

Here's why I think it's worth knowing about: the GKSL form is a *constrained parametrization*. When you write your belief dynamics in this form, positivity and normalization are *structural*---they hold automatically, not because you've carefully tuned things. And you get a clean separation between "conservative prediction" (reversible internal rollouts) and "dissipative grounding" (irreversible assimilation of boundary information).

Think of it as an elegant way to write down belief dynamics that are guaranteed to be well-behaved. You don't have to use it, but it's good to know it exists.
:::

:::{prf:definition} Belief operator
:label: def-belief-operator

Let $\varrho_t\in\mathbb{C}^{d\times d}$ satisfy $\varrho_t\succeq 0$ and $\mathrm{Tr}(\varrho_t)=1$. Diagonal $\varrho_t$ reduces to a classical probability vector; non-diagonal terms can be used to encode correlations/uncertainty structure in a learned feature basis.

:::

:::{div} feynman-prose
The definition above says: instead of representing belief as a vector $p \in \mathbb{R}^n$, represent it as a matrix $\varrho \in \mathbb{C}^{d \times d}$. Why would you do this? Because a matrix can encode *more* than just marginal probabilities---the off-diagonal terms can represent correlations, coherences, or structured uncertainty. If you only want classical probabilities, use a diagonal matrix and you're back to a vector.
:::

:::{prf:definition} GKSL generator
:label: def-gksl-generator

A continuous-time, Markovian, completely-positive trace-preserving (CPTP) evolution has a generator of the Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) form {cite}`gorini1976completely,lindblad1976generators`:

$$
\frac{d\varrho}{ds}
=
\underbrace{-i[H,\varrho]}_{\text{conservative drift}}
\;+\;
\underbrace{\sum_{j} \gamma_j\left(L_j\varrho L_j^\dagger-\frac12\{L_j^\dagger L_j,\varrho\}\right)}_{\text{dissipative update}},

$$
where {math}`H=H^\dagger` is Hermitian, {math}`\gamma_j\ge 0` are rates, and {math}`\{L_j\}` are (learned) operators.

**Operational interpretation (within this document).**
- The commutator term is a structured way to represent **reversible internal prediction** (it preserves $\mathrm{Tr}(\varrho)$ and the spectrum of $\varrho$).
- The dissipator is a structured way to represent **irreversible assimilation / disturbance** while preserving positivity and trace.

This is a modeling choice, not a claim about literal quantum physics: it is used here purely as a convenient, well-posed parametrization of CPTP belief updates.

*Note (WFR Correspondence).* In the **classical limit** (diagonal density matrix $\varrho = \mathrm{diag}(p)$), the GKSL generator reduces to a Markov jump process on the diagonal probabilities $p_k$. This classical master equation is **rigorously equivalent** to a gradient flow in the Wasserstein-Fisher-Rao metric ({prf:ref}`def-the-wfr-action`, {ref}`sec-connection-to-gksl-master-equation`): transport corresponds to continuous probability flow, reaction corresponds to jump-induced mass redistribution {cite}`maas2011gradient,mielke2011gradient`. The commutator term vanishes for diagonal states (no coherences to rotate). For full quantum states, see {cite}`carlen2014wasserstein` for the quantum Wasserstein gradient flow theory.

:::

:::{div} feynman-prose
Let me unpack that equation because it has a beautiful structure:

**The commutator term** $-i[H, \varrho]$ is the "conservative" part. If this were the whole equation, belief would evolve *reversibly*---like a Hamiltonian system rolling forward. Nothing is created or destroyed; structure is preserved. This is your internal simulation running forward in its own head.

**The dissipator term** is the "irreversible" part. The operators $L_j$ represent different kinds of "disturbances" or "jumps" that can happen. Each one has a rate $\gamma_j$. This is where boundary information enters---where reality pokes holes in your internal model and forces corrections.

The magic is that this decomposition is *complete*: any Markovian, completely-positive, trace-preserving (CPTP) evolution can be written this way. So you're not restricting what dynamics are possible; you're just organizing them into "reversible" and "irreversible" buckets.
:::

(pi-lindblad)=
::::{admonition} Physics Isomorphism: Lindblad Master Equation
:class: note

**In Physics:** The GKSL (Gorini-Kossakowski-Sudarshan-Lindblad) equation describes the evolution of open quantum systems: $\dot{\varrho} = -i[H,\varrho] + \sum_k \gamma_k(L_k\varrho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \varrho\})$. It is the most general Markovian, completely positive, trace-preserving (CPTP) evolution {cite}`lindblad1976generators,gorini1976completely`.

**In Implementation:** The belief density evolution (Definition {prf:ref}`def-gksl-generator`):

$$
\mathcal{L}_{\text{GKSL}}(\varrho) = -i[H_{\text{eff}}, \varrho] + \sum_k \gamma_k \left( L_k \varrho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \varrho\} \right)

$$
**Correspondence Table:**
| Open Quantum Systems | Agent (Belief Dynamics) |
|:---------------------|:------------------------|
| Density matrix $\varrho$ | Belief distribution $\rho$ |
| Hamiltonian $H$ | Effective potential $\Phi_{\text{eff}}$ ({prf:ref}`def-effective-potential`) |
| Lindblad operators $L_k$ | Jump operators (chart transitions) |
| Decoherence rate $\gamma_k$ | Transition rates |
| CPTP evolution | Probability-preserving dynamics |

**Diagnostic:** MECCheck (Node 22) monitors $\|\dot{\varrho} - \mathcal{L}_{\text{GKSL}}(\varrho)\|_F^2$.
::::

(sec-master-equation-consistency-defect)=
### Master-Equation Consistency Defect (Node 22)

:::{div} feynman-prose
Now here's where the rubber meets the road. We have this beautiful GKSL form that tells us what a *consistent* belief update should look like. The actual agent is doing some update---Bayesian filtering, Sieve projection, whatever. How do we know if the actual update is consistent with the GKSL template?

We compare them. The **consistency defect** is simply the squared Frobenius norm of the difference between what the agent actually did and what the GKSL equation predicts. If this is small, the agent's belief dynamics are well-behaved. If it's large, something is wrong---maybe the agent is updating too aggressively, or the parametrization is missing important structure.
:::

:::{div} feynman-prose
If an implementation maintains an operator belief $\varrho_t$ and produces an empirical update $\varrho_{t+1}$ (e.g., after a boundary update + Sieve projection), then a **consistency defect** compares it to the GKSL-predicted infinitesimal update:

$$
\mathcal{L}_{\text{MEC}}
:=
\left\|
\frac{\varrho_{t+1}-\varrho_t}{\Delta t}
\;-\;
\mathcal{L}_{\text{GKSL}}(\varrho_t)
\right\|_F^2,

$$
where $\mathcal{L}_{\text{GKSL}}(\cdot)$ denotes the right-hand side of {prf:ref}`def-gksl-generator`. This is the quantity monitored by MECCheck (Node 22).
:::

(sec-residual-event-codebook)=
### Residual-Event ("Jump") Codebook (Links to {ref}`sec-defect-functionals-implementing-regulation`.B)

:::{div} feynman-prose
Here's a practical question: if we want to use the GKSL form, we need to specify those $L_j$ operators---the different "types of disturbances" that can happen. Where do they come from?

One answer: learn them from data. The idea is to build a **codebook** of disturbance types, just like a VQ-VAE builds a codebook of image patches. When something unexpected happens---when the world deviates from your model's prediction---you classify that deviation into one of your disturbance types. This gives you a discrete label ("what kind of surprise was this?") that you can use to select the appropriate $L_j$.

The key insight is that we should attach this codebook to the **structured nuisance** channel, not to texture. Texture is reconstruction detail; it doesn't drive macro dynamics. Nuisance is structured variation that affects how the world evolves. The disturbance library should capture patterns in nuisance residuals, not patterns in texture residuals.
:::

:::{div} feynman-prose
The GKSL form becomes implementable if we can parameterize a *finite* family of disturbance/update types. With the nuisance/texture split ({ref}`sec-the-shutter-as-a-vq-vae`), the disturbance library should attach to the **structured nuisance** channel, not to texture. A practical route is a discrete codebook over one-step nuisance residuals:
1. Compute a one-step prediction $(k_{t+1}^{\text{pred}}, z_{n,t+1}^{\text{pred}}):=S(K_t,z_{n,t},a_t)$ from the world model (macro + nuisance only).
2. Encode the next observation to obtain $(K_{t+1}, z_{n,t+1}, z_{\text{tex},t+1})$ via the shutter.
3. Form the **nuisance residual** $\Delta z_{n,t}:=z_{n,t+1}-z_{n,t+1}^{\text{pred}}$.
4. Quantize $\Delta z_{n,t}$ with a second VQ module to obtain $J_t\in\{1,\dots,|\mathcal{J}|\}$.

Texture $z_{\text{tex}}$ is treated as an emission/likelihood residual: it is used to model $p(x_t\mid K_t,z_{n,t},z_{\text{tex},t})$ but is not used to define jump types. This is the formal reconciliation: "jumps" model **structured disturbances**, while "texture" models **measurement detail**.
:::

:::{admonition} Two Uses for the Disturbance Codebook
:class: feynman-added note

The index $J_t$ can be used in two ways:
- **Classical residual modeling:** store representative nuisance residual vectors (or residual distributions) per code and train a conditional noise model $p(\Delta z_n\mid J)$.
- **Operator-valued modeling (optional):** associate each residual code $j$ with a learned low-rank operator $L_j$ and let rates $\gamma_j$ be predicted online; this is the operator analogue of a mixture-of-disturbances model.

The core engineering benefit is identifiability: the agent exposes a discrete label for "what kind of unmodeled disturbance happened", rather than forcing the macro register to absorb it.
:::

(sec-update-vs-evidence-check-and-metric-speed-limit)=
### Update vs Evidence Check (Node 23) and Metric Speed Limit (Node 24)

:::{div} feynman-prose
Even if you don't want to go full operator-valued beliefs, you can still monitor the "no free update" principle. The idea is simple: your beliefs shouldn't change faster than your observations justify.

Think of it this way. You're receiving a certain amount of information from the boundary per timestep---call it $I(X_t; K_t)$, the mutual information between observations and macro-states. That's like a budget. Your belief update should not "spend" more than you have. If your beliefs are changing by more KL-divergence than you're receiving in mutual information, you're hallucinating---updating based on internal fantasies rather than external evidence.

Similarly, there's a speed limit on how fast your internal state can move. If $z_t$ is jumping around wildly from step to step, something is wrong. Either your representation is unstable, or your updates are too aggressive. The metric speed limit says: "under the geometry of your state space, don't move faster than $v_{\max}$ per step."
:::

:::{div} feynman-prose
Even without operator beliefs, the same "no free update" principle can be monitored in classical terms:
:::

:::{admonition} Update vs Evidence (NEPCheck)
:class: feynman-added note

Penalize belief updates that change faster than boundary information supports:

$$
\mathcal{L}_{\text{NEP}}
:=
\mathrm{ReLU}\!\left(D_{\mathrm{KL}}(p_{t+1}\Vert p_t)-I(X_t;K_t)\right)^2.

$$

**In plain words:** "The KL-divergence from old belief to new belief is how much your mind changed. The mutual information $I(X_t; K_t)$ is how much evidence you received. If you changed your mind more than your evidence justified, that's a problem."

This is a conservative audit metric: it does not assert a physical entropy law, but it detects ungrounded internal updating relative to measured boundary coupling (Node 13).
:::

:::{admonition} Metric Speed Limit (QSLCheck)
:class: feynman-added note

Impose a hard/soft bound on how far internal state may move per step under the state-space metric:

$$
\mathcal{L}_{\text{QSL}}:=\mathrm{ReLU}\!\left(d_G(z_{t+1},z_t)-v_{\max}\right)^2,

$$

**In plain words:** "Measure the distance traveled in state space using the metric $G$. If it exceeds the speed limit $v_{\max}$, penalize."

This is a geometry-consistent generalization of KL-per-update constraints (ZenoCheck).
:::

::::{admonition} Connection to RL #19: POMDP Belief Updates as Degenerate Belief Dynamics
:class: note
:name: conn-rl-19
**The General Law (Fragile Agent):**
Belief evolution follows the **Filtering + Projection Template** on the discrete macro register:

$$
p_{t+1}(k') = \frac{L_{t+1}(k')\, \tilde{p}_{t+1}(k')}{\sum_j L_{t+1}(j)\, \tilde{p}_{t+1}(j)}, \quad \tilde{p}_{t+1}(k') = \sum_k p_t(k)\, \bar{P}(k'|k,a_t)

$$
with **Sieve projections** applied after each update: hard masking or soft reweighting to enforce feasibility constraints.

**The Degenerate Limit:**
Remove the Sieve projections ($\text{feasible}(k) = 1$ for all $k$). Use continuous beliefs without discrete macro-register.

**The Special Case (Standard RL):**

$$
b_{t+1}(s') \propto O(o_{t+1}|s') \sum_s T(s'|s,a) b_t(s)

$$
This recovers standard **POMDP belief updates** {cite}`kaelbling1998planning` without safety constraints.

**What the generalization offers:**
- **Safety-aware beliefs**: Sieve projections ({ref}`sec-sieve-events-as-projections-reweightings`) remove probability mass from unsafe states *before* action selection
- **Discrete auditable symbols**: $H(K) \le \log|\mathcal{K}|$ provides hard capacity bound; standard POMDPs have unbounded continuous beliefs
- **Constraint enforcement**: Gate Nodes trigger belief reweighting when diagnostics fail (NEPCheck, QSLCheck)
- **Operator-valued updates**: {ref}`sec-optional-operator-valued-belief-updates` extends to GKSL/Lindblad form for quantum-like belief decoherence
::::



(sec-correspondence-table-filtering-control-template)=
## Correspondence Table: Filtering / Control Template

:::{div} feynman-prose
Let me close with a translation dictionary. If you're coming from filtering/control theory, here's how to map your vocabulary to the Fragile Agent components. The key point: it's all the same math, just organized with safety constraints made explicit.
:::

:::{div} feynman-added
The table below is a dictionary from standard **filtering and constrained inference** to the Fragile Agent components. It is purely classical: belief evolution is "predict - update - project".

| Filtering / Control Object                                | Fragile Agent Equivalent                       | Role                          |
|:----------------------------------------------------------|:-----------------------------------------------|:------------------------------|
| Belief state $p_t(k)$                                     | Macro belief over $\mathcal{K}$                | Summary statistic for control |
| Prediction $\tilde p_{t+1}=\bar{P}^\top p_t$              | Macro dynamics model $\bar{P}(k'\mid k,a)$     | One-step forecast             |
| Likelihood $L_{t+1}(k)=p(x_{t+1}\mid k)$                  | Shutter/emission score for macrostates         | Boundary grounding signal     |
| Bayes update $p_{t+1}\propto L_{t+1}\odot \tilde p_{t+1}$ | Assimilation step                              | Incorporate observations      |
| Projection / reweighting $p'_t$                           | Sieve checks (CostBoundCheck, CompactCheck, ...) | Enforce feasibility/stability |
| Entropy $H(p_t)$                                          | Macro uncertainty / symbol mixing              | Detect collapse vs dispersion |
| KL-control $D_{\mathrm{KL}}(\pi\Vert\pi_0)$               | Control-effort regularizer                     | Penalize deviation from prior |
:::
