(sec-failure-modes)=
# Failure Modes (Observed Pathologies)

## TLDR

- Enumerate a “periodic table” of **observable failure patterns** (collapse, oscillation, Zeno/chatter, overfitting,
  paralysis, fragility) and map each to a responsible component.
- Treat failures as **structured diagnostics**: identify the signature, then apply the corresponding intervention
  (damping, regularization, projection, reset, curriculum changes).
- Use the table to distinguish **fundamental limits** (barriers) from **implementation/optimization issues**
  (fixable by schedule/architecture changes).
- Provide a shared vocabulary for debugging: “what failed” is named, and “where it lives” is explicit.
- This chapter pairs with `Diagnostics` and `Barriers`: together they define what to monitor, what limits exist, and what
  to do when something triggers.

## Roadmap

1. Failure taxonomy and how to read the table.
2. Canonical interventions and what signals should trigger them.
3. Worked interpretations: how failures relate to Sieve nodes and barrier surfaces.

:::{div} feynman-prose
Here is something fascinating about learning systems: they fail in characteristic ways. Not randomly, but in patterns that repeat across wildly different domains. Whether you are training a robot to walk or a neural network to recognize faces, the same breakdowns keep appearing.

Why? Because every learning system faces the same fundamental challenges: balance exploration against exploitation, compress the world into internal representations, and update behavior based on feedback. When any of these goes wrong, you get a universal failure pattern.

What follows is a "periodic table" of failure modes. Just as chemists organized elements by their properties and could predict how unknown elements would behave, we can organize learning failures by their signatures and predict how to fix them. The key insight: each failure traces back to a specific component, and that component tells us exactly where to look for the repair.
:::

(rb-rl-pathologies)=
:::{admonition} Researcher Bridge: RL Pathologies, Named and Localized
:class: info
If you have seen mode collapse, oscillation, overfitting, or deadlock in RL, this table is the same landscape but made explicit. Each failure is tied to a component and a diagnostic signature, so it can be detected and corrected rather than discovered post hoc.
:::

:::{div} feynman-prose
Before diving into the table, let me give you a feel for its structure.

Each mode has a two-letter code: coordinates in "failure space." The first letter tells you the *type* of dynamics going wrong (D for dispersion, C for concentration, T for topology), and the second tells you the *direction* (spreading out, collapsing inward, or stuck).

The "Failed Component" column is crucial. It tells you where the disease lives. Is it in the Policy? The Shutter? The World Model? The Critic? Knowing this immediately narrows your debugging search.

Here is the beautiful thing: once you know the component and failure type, the intervention almost writes itself. A policy that oscillates needs damping. A world model that overfits needs regularization. Like medicine, the diagnosis determines the treatment.
:::

When Limits are breached or Interfaces fail, the agent exhibits specific pathologies.

| Mode    | Standard Name       | Failed Component     | Fragile (Pathology) Name      | Description                                                                     |
|---------|---------------------|----------------------|-------------------------------|---------------------------------------------------------------------------------|
| **D.D** | Dispersion-Decay    | **All (Optimal)**    | **Success (Convergence)**     | Agent solves task; error drops to a stable floor.                               |
| **S.E** | Subcritical-Equilib | **Policy**           | **Curriculum Stumble**        | Task difficulty increases faster than adaptation rate.                          |
| **C.D** | Conc-Dispersion     | **Policy/Shutter**   | **Mode Collapse / Obsession** | Policy concentrates on a single mode, neglecting remaining state space.         |
| **C.E** | Conc-Escape         | **Policy/Critic**    | **Divergence / Blow-up**      | Gradients/activations diverge; optimization becomes unstable.                   |
| **T.E** | Topo-Extension      | **Shutter/WM**       | **Wrong Paradigm**            | Architecture is topologically insufficient.                                     |
| **S.D** | Struct-Dispersion   | **Shutter**          | **Symmetry Blindness**        | Fails to exploit available symmetries.                                          |
| **C.C** | Event Accumulation  | **Policy/WM**        | **Decision Paralysis**        | Input happens faster than decision loop (Zeno).                                 |
| **T.D** | Glassy Freeze       | **Policy**           | **Learned Helplessness**      | Policy converges to suboptimal fixed point with zero gradient.                  |
| **D.E** | Oscillatory         | **Policy**           | **Pilot-Induced Oscillation** | Overcorrection causes increasing instability.                                   |
| **T.C** | Labyrinthine        | **World Model**      | **Overfitting to Noise**      | WM models noise instead of signal.                                              |
| **D.C** | Semantic Horizon    | **Shutter/WM**       | **Ungrounded inference**      | Distribution shift causes internal rollouts to decouple from boundary evidence. |
| **B.E** | Sensitivity Expl.   | **Critic**           | **Fragility**                 | Optimization for a single condition induces high sensitivity to perturbations.  |
| **B.D** | Resource Depletion  | **Boundary/Shutter** | **Starvation**                | Input or power resources depleted.                                              |
| **B.C** | Control Deficit     | **Policy**           | **Overwhelmed**               | Disturbance more complex than controller (Ashby).                               |

:::{div} feynman-prose
A few of these deserve special attention.

**D.D (Success)** is not a failure at all. It is what happens when everything works: errors decay to a stable floor. I include it as the reference point. When debugging, you need to know what "healthy" looks like.

**C.D (Mode Collapse)** is among the most common failures in deep learning. The policy becomes obsessed with one solution and ignores everything else. Picture a robot that learns to stand still because standing still is safe, even though its task is to walk. Probability mass concentrates on one mode; exploration vanishes.

**T.E (Wrong Paradigm)** is the most subtle failure. The system is not learning poorly; it *cannot* represent the solution. This is like trying to model a spiral staircase with a flat piece of paper. No matter how cleverly you fold it, the topology is wrong. You need to change the architecture, not tune the parameters.

**D.E (Pilot-Induced Oscillation)** is the opposite of helplessness. The agent is *too* responsive, overcorrecting each error and making things worse. Pilots call this "PIO," and it crashes airplanes. The cure is not more learning but more damping.
:::

(sec-interventions)=
## Interventions (Mitigations)

:::{div} feynman-prose
Now the practical part: what do you *do* when things go wrong?

Interventions are not magic. Each is a targeted surgery for a specific pathology. You would not give antibiotics for a broken bone, and you would not apply gradient clipping to fix mode collapse. The failure mode determines the intervention.

Think of these as a doctor's toolkit. When diagnostics detect a failure signature, they prescribe the corresponding treatment. This differs from the usual ML approach of randomly trying tricks until something works. Here, we are systematic: diagnosis first, then treatment.
:::

(rb-heuristic-fixes)=
:::{admonition} Researcher Bridge: Heuristic Fixes as Typed Surgeries
:class: tip
These interventions correspond mathematically to common RL stabilizers: target networks, clipping, entropy tuning, replay, and resets. Each intervention is triggered by a specific diagnostic condition rather than manual hyperparameter tuning.
:::

:::{div} feynman-prose
Here is how to read the intervention table.

**Surgery ID**: Each intervention is named "Surg" plus the failure mode code. SurgCE is the surgery for C.E (Divergence).

**Target Mode**: Which failure triggers this intervention. When diagnostics see this failure, consider this surgery.

**Target Component**: Which part of the system you are operating on. Change the minimum necessary to fix the problem.

**Fragile (Upgrade) Translation**: The conceptual category. Is it a limiter? A reset? A regularizer? This tells you the *type* of intervention.

**Mechanism**: The actual implementation, what to change and how.
:::

Interventions are external mitigations to restore stability, re-ground the representation, or reduce unsafe update rates.

| Surgery ID     | Target Mode        | Target Component     | Fragile (Upgrade) Translation | Mechanism                                                                                                                                                                                                                                                                                                                               |
|----------------|--------------------|----------------------|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **SurgCE**     | C.E (Divergence)   | **Policy/Critic**    | **Limiter / trust region**    | **Gradient Clipping / Trust Region:** Clamp outputs; enforce $\Vert \pi_{new} - \pi_{old} \Vert < \delta$.                                                                                                                                                                                                                              |
| **SurgCC**     | C.C (Zeno)         | **WM/Policy**        | **Time-boxing / Rate Limit**  | **Skip-Frame / Latency:** Force fixed $\Delta t$; ignore inputs during cool-down.                                                                                                                                                                                                                                                       |
| **SurgCD_Alt** | C.D (Obsession)    | **Policy**           | **Reset / Reshuffling**       | **Re-initialization:** Reset parameters of the obsession-locked sub-module to random.                                                                                                                                                                                                                                                   |
| **SurgSE**     | S.E (Stumble)      | **World Model**      | **Curriculum Ease-off**       | **Curriculum Learning:** Reduce Task Difficulty or Rewind to earlier level.                                                                                                                                                                                                                                                             |
| **SurgSC**     | S.C (Instability)  | **Critic**           | **Parameter Freezing**        | **Target Network Freeze:** Stop updating Target V; switch to slower exponential moving average.                                                                                                                                                                                                                                         |
| **SurgCD**     | C.D (Collapse)     | **Shutter**          | **Feature Pruning**           | **Dead Code Pruning:** Identify and excise unused macro symbols / dead fibres.                                                                                                                                                                                                                                                          |
| **SurgSD**     | S.D (Blindness)    | **Shutter**          | **Augmentation / Ghost Vars** | **Domain Randomization:** Inject noise into $x$ to force the shutter to learn robust macrostates.                                                                                                                                                                                                                                       |
| **SurgTE**     | T.E (Paradigm)     | **Shutter/WM**       | **Architecture Search**       | **Neural Architecture Search (NAS):** Modify shutter+WM class to match topology (e.g., add hierarchy / memory).                                                                                                                                                                                                                         |
| **SurgTC**     | T.C (Overfit)      | **WM**               | **Regularization**            | **Weight Decay / Dropout:** Increase $\lambda \lVert\theta\rVert^2$ penalty.                                                                                                                                                                                                                                                            |
| **SurgTD**     | T.D (Helplessness) | **Policy**           | **Noise Injection**           | **Parameter Space Noise:** Add $\xi \sim \mathcal{N}(0, \Sigma)$ to Policy weights.                                                                                                                                                                                                                                                     |
| **SurgDC**     | D.C (Ungrounded)   | **Shutter/WM**       | **Smoothing / fallback**      | **OOD rejection:** if nuisance surprisal spikes (e.g. $D_{\mathrm{KL}}(q(z_n\mid x)\Vert p(z_n))>\tau_n$) and/or texture surprisal spikes (e.g. $D_{\mathrm{KL}}(q(z_{\mathrm{tex}}\mid x)\Vert p(z_{\mathrm{tex}}))>\tau_{\mathrm{tex}}$) and/or macro surprisal spikes (e.g. $-\log p_\psi(K)>\tau_K$), trigger fallback (safe stop). |
| **SurgDE**     | D.E (Oscillate)    | **Policy**           | **Damping**                   | **Triggered by OscillateCheck / HolonomyCheck:** reduce policy step size (lower LR), decrease Adam $\beta_1$, increase batch size, or temporarily freeze policy updates until the critic signal is stable.                                                                                                                              |
| **SurgBE**     | B.E (Fragile)      | **Critic**           | **Saturation / Anti-Windup**  | **Spectral Normalization:** Constrain Lipschitz constant of $V(z)$.                                                                                                                                                                                                                                                                     |
| **SurgBD**     | B.D (Starve)       | **Boundary/Shutter** | **Replay Buffer / Reservoir** | **Experience Replay:** Train on historical buffers to prevent catastrophic forgetting.                                                                                                                                                                                                                                                  |
| **SurgBC**     | B.C (Deficit)      | **Policy**           | **Controller Expansion**      | **Width Expansion:** Dynamically add neurons to the Policy network (Net2Net).                                                                                                                                                                                                                                                           |

:::{div} feynman-prose
Several patterns in these interventions are worth noticing.

First, many interventions involve *slowing down* or *constraining* the system. SurgCE limits how far the policy can move. SurgCC enforces time-boxing. SurgSC freezes target networks. SurgDE reduces learning rates. This reflects a deep principle: when a dynamical system is unstable, try damping first. Let it settle before pushing further.

Second, several interventions involve *adding noise*. SurgTD adds parameter noise. SurgSD adds domain randomization. SurgCD_Alt resets parameters randomly. This seems counterintuitive: why would chaos help a struggling system? Because these failures (helplessness, blindness, obsession) are cases where the system is trapped in a local minimum or has lost diversity. Noise is the universal escape mechanism.

Third, consider SurgDC (Out-of-Distribution rejection). This is the most sophisticated intervention: the system must recognize when it is out of its depth. When surprisal on nuisance factors or texture spikes, stop trusting the internal model and fall back to safe behavior. This is epistemic humility encoded into the control loop.

Finally, only SurgBC (Controller Expansion) and SurgTE (Architecture Search) actually grow the system. All others work within the existing architecture. Most problems can be fixed without architectural changes, but some genuinely require more capacity. The diagnostic system helps you distinguish these cases.
:::

:::{note}
:class: feynman-added
The relationship between failures and interventions is not always one-to-one. Some failures may require multiple interventions applied in sequence, and some interventions may help with multiple failure modes. The table gives the primary mapping, but clinical judgment is still required.
:::
