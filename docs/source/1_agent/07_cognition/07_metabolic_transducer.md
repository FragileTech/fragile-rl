(sec-the-metabolic-transducer-autopoiesis-and-the-szilard-engine)=
# The Metabolic Transducer: Autopoiesis and the Szilard Engine

## TLDR

- Close the thermodynamic loop: if computation dissipates energy (Landauer), then successful prediction/control must
  enable **work extraction** (Szilard-engine view).
- Define the **Metabolic Transducer** as the mechanism turning reward/information into usable metabolic budget.
- Derive the **autopoietic viability condition**: sustained agency requires harvest rate to exceed dissipation.
- Show that geometry itself degrades when energy is depleted (a “fading metric law”), yielding concrete failure modes.
- Introduce diagnostics that monitor metabolic viability and couple directly to the Governor and parameter sieve.

## Roadmap

1. Information harvesting (Szilard engine) and the meaning of reward as work.
2. Autopoietic inequality and survival/viability constraints.
3. Geometry under depletion + runtime diagnostics for metabolic failure.

*Abstract.* This chapter closes the thermodynamic loop opened in
{ref}`sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics`. We derive the
**Metabolic Transducer** $\mathfrak{T}_{\text{harvest}}$ from the Szilard engine analysis, showing that reward signals
encode extractable work. We prove the **Autopoietic Inequality**—the survival condition requiring harvest rate to exceed
metabolic dissipation. We derive the **Fading Metric Law** from Fisher Information principles, showing that the latent
geometry degrades as energy depletes. Finally, we introduce diagnostic nodes 67–70 to monitor autopoietic viability.

*Cross-references:*
- Closes the loop from {ref}`sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics`
  (Metabolism).
- Connects the Reward Field ({ref}`sec-the-reward-field-value-forms-and-hodge-geometry`) to survival.
- Provides the ultimate constraint for the Universal Governor
  ({ref}`sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller`).
- Adds metabolic viability to the Parameter Space Sieve ({ref}`sec-parameter-space-sieve`).

*Literature:*
- Maxwell's Demon {cite}`maxwell1871theory` (Maxwell, *Theory of Heat*, 1871).
- Szilard engine {cite}`szilard1929entropy` (Szilard, "On the decrease of entropy in a thermodynamic system by the
  intervention of intelligent beings," *Zeitschrift für Physik* 53:840–856, 1929).
- Landauer's principle {cite}`landauer1961irreversibility` (Landauer, "Irreversibility and Heat Generation in the
  Computing Process," *IBM J. Res. Dev.* 5:183–191, 1961).
- Autopoiesis {cite}`maturana1980autopoiesis` (Maturana & Varela, *Autopoiesis and Cognition: The Realization of the
  Living*, 1980).
- Free energy principle {cite}`friston2010free` (Friston, "The free-energy principle: a unified brain theory?",
  *Nature Reviews Neuroscience* 11:127–138, 2010).
- Johnson–Nyquist noise {cite}`johnson1928thermal,nyquist1928thermal` (Johnson, "Thermal Agitation of Electricity in
  Conductors," *Phys. Rev.* 32:97–109, 1928; Nyquist, "Thermal Agitation of Electric Charge in Conductors,"
  *Phys. Rev.* 32:110–113, 1928).

:::{div} feynman-prose
Now we come to something rather beautiful. In the last chapter, we showed that thinking costs energy—the Landauer bound tells us that every time you sharpen your beliefs, you have to pay a thermodynamic price. But that was only half the story. It is like learning that running costs calories without ever mentioning that eating provides them.

Here is the question we must answer: **Where does the energy come from?**

The answer turns out to be deeply connected to one of the most famous puzzles in physics—Maxwell's Demon. And when we work it through, we will discover something remarkable: the reward signal in reinforcement learning is not just an arbitrary training signal. It is information about extractable work. Getting a positive reward means you have found yourself in a configuration where you can harvest energy from the environment.

This chapter will close the thermodynamic loop. We will show that an agent is really a kind of engine—a machine that converts information about low-entropy configurations into the energy needed to sustain its own computational processes. The survival condition for such a machine is almost trivially simple to state: **you must harvest more than you burn**. But the consequences of this simple statement are profound.
:::



(sec-thermodynamics-of-information-harvesting)=
## The Thermodynamics of Information Harvesting

In {ref}`sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics`, we established that computation dissipates energy: the Generalized Landauer Bound (Theorem {prf:ref}`thm-generalized-landauer-bound`) states $\dot{\mathcal{M}} \geq T_c |dH/ds|$. Here we establish the converse: **correct prediction extracts work**. This is the Szilard engine operating in the forward direction.

:::{div} feynman-prose
Let me explain what we are after. In the metabolism chapter, we proved that sharpening your beliefs—reducing uncertainty—costs energy. That is Landauer's principle in action. But Landauer's principle has a flip side, and it is this flip side that makes life possible.

Think about it this way. When you reduce entropy internally, you have to dump that entropy somewhere. But the converse is also true: if you *know* something about a system—if you have information about where a molecule is, or where the food is, or what the stock market will do tomorrow—you can use that information to extract work.

This is exactly what Leo Szilard figured out in 1929, and it is the key to understanding how agents can sustain themselves.
:::

:::{prf:definition} The Reward Flux
:label: def-reward-flux-harvesting

The **Reward Flux** $J_r(t)$ is the instantaneous rate of reward accumulation (Definition {prf:ref}`def-the-reward-flux`):

$$
J_r(t) = \langle \mathcal{R}(z_t), v_t \rangle_G = r_t

$$

where $\mathcal{R}$ is the reward 1-form ({ref}`sec-the-reward-field-value-forms-and-hodge-geometry`) and $v_t = \dot{z}_t$ is the velocity in latent space.

*Units:* $[J_r] = \text{nats/step}$ (information-theoretic) or $[\text{utility/step}]$ (decision-theoretic).

*Interpretation:* A positive reward $r_t > 0$ indicates the agent has navigated to a state with lower environmental entropy—a configuration where resources (food, fuel, safety) are localized and accessible.

:::

:::{div} feynman-prose
What does "lower environmental entropy" mean in practice? Think of it this way. The universe tends toward disorder—that is the Second Law. A high-entropy configuration is one where everything is mixed up, spread out, uniform. Low entropy means structure, organization, resources concentrated where you can use them.

When you find food, you have found a low-entropy configuration: calories packed into a small volume instead of spread uniformly across the landscape. When you solve a problem, you have found a low-entropy configuration: the answer isolated from the vast space of wrong answers. The reward signal is telling you: "You found structure. You found order. You found something useful."

And here is the crucial point: **information about low-entropy configurations is equivalent to extractable work**.
:::

:::{prf:definition} Information Utility
:label: def-information-utility

The **Information Utility** $\mathcal{I}_{\text{util}}(r_t)$ quantifies the actionable information content of the reward signal:

$$
\mathcal{I}_{\text{util}}(r_t) := I(Z_t; R_t) = H[R_t] - H[R_t \mid Z_t]

$$

where $I(Z_t; R_t)$ is the mutual information between the agent's state $Z_t$ and the reward $R_t$.

*Operational interpretation:* This is the reduction in uncertainty about environmental resources achieved by navigating to state $z_t$ and observing reward $r_t$.

*Units:* $[\mathcal{I}_{\text{util}}] = \text{nats}$ (or bits if using $\log_2$).

*Simplification:* When the reward signal is deterministic given state, $H[R_t \mid Z_t] = 0$, so $\mathcal{I}_{\text{util}}(r_t) = H[R_t]$. In practice, we often use the approximation $\mathcal{I}_{\text{util}}(r_t) \approx |r_t|$ for rewards measured in natural units.

:::

:::{prf:axiom} The Szilard Correspondence (Information-Work Duality)
:label: ax-szilard-correspondence

Information about low-entropy configurations can be converted to extractable work. Specifically, if an agent possesses $I$ nats of mutual information with a thermal reservoir at temperature $T_{\text{env}}$, it can extract at most:

$$
W_{\max} = k_B T_{\text{env}} \cdot I

$$

joules of work, where $k_B$ is Boltzmann's constant.

*Physical basis:* This is the inverse of Landauer's principle. Landauer states that erasing 1 bit costs $k_B T \ln 2$ joules. Szilard's engine demonstrates that acquiring 1 bit about a system enables extracting $k_B T \ln 2$ joules. The two are thermodynamically dual.

*Cognitive interpretation:* A reward signal $r_t > 0$ encodes mutual information between the agent's state and resource availability. This information, when acted upon, enables work extraction from the environment.

:::

:::{div} feynman-prose
This axiom is the heart of the whole chapter, so let me make sure you really understand it.

Landauer says: to erase a bit, you must pay $k_B T \ln 2$ joules. That is the cost of forgetting.

Szilard says: if you *know* a bit about a thermal system, you can *extract* $k_B T \ln 2$ joules. That is the profit of knowing.

These are not separate facts—they are two sides of the same coin. The universe has a kind of accounting system, and information is the currency. You cannot get something for nothing. But crucially, you cannot lose something for nothing either. Information has value, and that value can be converted to work.

Now, what does this mean for an agent navigating the world? When the agent receives a positive reward, it means: "You have located something valuable. You have acquired information about where resources are." And that information—*that very information*—can be converted to extractable work at the rate $k_B T$ per nat.

This is not a metaphor. It is a thermodynamic fact.
:::

:::{admonition} Example: The Szilard Engine Step by Step
:class: feynman-added example

Let us trace through the Szilard engine carefully, because it is one of those arguments where if you miss a step, the whole thing seems like magic.

**Setup:** A single molecule of gas in a box at temperature $T$. The molecule is bouncing around, and we do not know which half of the box it is in.

**Step 1 (Measurement):** A "demon" measures which half the molecule occupies. This measurement gives 1 bit of information. But here is the key: the demon must store this information somewhere, and by Landauer's principle, eventually erasing this record will cost $k_B T \ln 2$ joules.

**Step 2 (Insertion):** The demon inserts a partition at the middle of the box. This costs negligible work if done slowly.

**Step 3 (Expansion):** Knowing which side the molecule is on, the demon lets the gas expand isothermally against a piston on that side. The molecule pushes the piston, doing work:

$$
W = \int P \, dV = \int \frac{k_B T}{V} dV = k_B T \ln 2

$$

**The Bottom Line:** The demon extracted $k_B T \ln 2$ joules of work, but it acquired 1 bit of information to do so. When that bit is eventually erased, the books balance. The Second Law survives.

**The Agent Version:** When your agent finds food (positive reward), it has acquired information about resource location. This information enables work extraction—consuming the food, charging the battery, sustaining computation.
:::

:::{prf:theorem} The Transducer Bound
:label: thm-szilard-transducer-bound

Let $r_t$ be the instantaneous reward signal with information content $\mathcal{I}_{\text{util}}(r_t)$ nats. The maximum free energy extractable per unit time is bounded by:

$$
\dot{E}_{\text{in}}^{\max}(t) = k_B T_{\text{env}} \cdot \mathcal{I}_{\text{util}}(r_t)

$$

where $T_{\text{env}}$ is the environmental temperature (characterizing energy availability).

*Proof sketch.*
1. The agent navigates to state $z_t$ and receives reward $r(z_t)$.
2. The reward encodes mutual information $I(Z_t; \text{Resource})$ between the agent's position and resource availability.
3. By the Szilard engine analysis, this mutual information enables extraction of $k_B T_{\text{env}} \cdot I$ joules.
4. The information utility $\mathcal{I}_{\text{util}}(r_t)$ quantifies the actionable information in the reward signal.
5. Real transduction incurs irreversibility losses captured by efficiency $\eta \leq 1$. $\square$

:::

:::{prf:definition} The Metabolic Transducer Operator
:label: def-metabolic-transducer

The **Metabolic Transducer** $\mathfrak{T}_{\text{harvest}}$ is the operator converting the reward flux to free energy flux:

$$
\dot{E}_{\text{in}}(t) = \mathfrak{T}_{\text{harvest}}(r_t) := \eta \cdot k_B T_{\text{env}} \cdot \mathcal{I}_{\text{util}}(r_t)

$$

where:
- $k_B \approx 1.38 \times 10^{-23}$ J/K is **Boltzmann's constant**
- $T_{\text{env}}$ is the **environmental temperature** (Kelvin)
- The product $k_B T_{\text{env}}$ is the **energy-per-nat conversion factor** (Joules/nat)
- $\eta \in [0, 1]$ is the **transduction efficiency** (Carnot-bounded, see Theorem {prf:ref}`thm-carnot-transduction-bound`)
- $\mathcal{I}_{\text{util}}(r_t)$ is the **information utility** of the reward signal (Definition {prf:ref}`def-information-utility`)

*Units:* $[\mathfrak{T}] = \text{Joules/step}$ (power).

*Simplified form:* For dimensionless analysis with $k_B = 1$, we write:

$$
\mathfrak{T}_{\text{harvest}}(r_t) = \eta \cdot T_{\text{env}} \cdot r_t

$$

where $r_t$ is measured in nats.

:::

:::{div} feynman-prose
The Metabolic Transducer is the agent's power plant. It takes in reward (information about low-entropy configurations) and outputs usable energy. The efficiency $\eta$ captures all the irreversibilities in the conversion process—nothing is perfect, and we will see later that $\eta$ is bounded by the Carnot limit.

Notice the beautiful symmetry with what we did in the metabolism chapter. There we had:

$$
\dot{\mathcal{M}} \geq T_c \left| \frac{dH}{ds} \right| \quad \text{(cost of sharpening beliefs)}

$$

And here we have:

$$
\dot{E}_{\text{in}} = \eta \cdot T_{\text{env}} \cdot r_t \quad \text{(income from correct predictions)}

$$

The agent lives in the gap between these two. If you can harvest more than you burn, you survive. If you cannot, you die. It really is that simple.
:::

::::{admonition} Physics Isomorphism: The Szilard Engine
:class: note
:name: pi-szilard-engine

**In Physics:** Leo Szilard (1929) resolved Maxwell's Demon paradox by showing that the demon must expend $k_B T \ln 2$ joules to measure a particle's position, thereby preserving the Second Law. Crucially, this implies the converse: if the demon *already knows* the particle's position (1 bit of information), it can extract $k_B T \ln 2$ joules of work by letting the particle expand isothermally against a piston.

**Derivation of the bound:**
1. Single-molecule gas in box of volume $V$ at temperature $T$
2. Demon measures which half the molecule occupies (1 bit = $\ln 2$ nats)
3. Demon inserts partition, molecule expands isothermally: $W = \int p \, dV = k_B T \ln 2$
4. Generalizing: $I$ nats of information enables $W = k_B T \cdot I$ joules extraction

**In Implementation:** The Metabolic Transducer $\mathfrak{T}$ implements this engine:
- **Measurement:** The agent spends $\dot{\mathcal{M}}$ to reduce belief entropy (locate resources)
- **Work Extraction:** Correct predictions enable harvesting reward $r_t$
- **Net Yield:** Survival requires $\mathfrak{T}(r_t) > \dot{\mathcal{M}}$

**Correspondence Table:**

| Thermodynamics | Fragile Agent | Symbol |
|:---------------|:--------------|:-------|
| Thermal reservoir | High-entropy environment | $T_{\text{env}}$ |
| Information acquisition | Perception/Inference | $\dot{\mathcal{M}}$ |
| Work extraction | Action/Harvesting | $\mathfrak{T}(r_t)$ |
| Stored work | Internal Battery | $B(t)$ |
| Second Law | Autopoietic Inequality | Theorem {prf:ref}`thm-autopoietic-inequality` |

::::



(sec-internal-battery-autopoietic-dynamics)=
## The Internal Battery and Autopoietic Dynamics

The agent maintains an **internal energy reservoir** that fuels computation. This reservoir is depleted by inference ({ref}`sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics`) and replenished by harvesting. The dynamics of this reservoir determine the agent's survival.

:::{div} feynman-prose
Now we need to talk about the agent's bank account. Not money—energy.

Every agent has some internal store of free energy. For a biological organism, this might be ATP molecules, glucose in the bloodstream, fat reserves. For an artificial agent, it might be battery charge or a compute budget. Whatever form it takes, this reservoir has a crucial property: **it can be depleted**.

And here is where things get existential. When the reservoir hits zero, the agent dies. Not metaphorically dies—actually stops functioning as a coherent computational system. The metric collapses, inference halts, and what was once an agent becomes a random walk in latent space.

The dynamics of this reservoir are simple bookkeeping: income minus expenses. But the consequences of this bookkeeping are profound.
:::

:::{prf:definition} The Internal Battery
:label: def-internal-battery

The **Internal Battery** $B(t)$ is a scalar state variable representing the agent's stored free energy:

$$
B: [0, \infty) \to [0, B_{\max}]

$$

where:
- $B_{\max}$ is the maximum storage capacity (Joules)
- $B(0) = B_0$ is the initial endowment

*Units:* $[B] = \text{Joules}$ (energy).

*Interpretation:* The battery represents the agent's capacity for future computation. In biological systems, this corresponds to ATP/glucose reserves; in artificial systems, to available compute budget.

:::

:::{prf:axiom} Energy Conservation (First Law)
:label: ax-energy-conservation-battery

The battery evolves according to the First Law of Thermodynamics:

$$
\frac{dB}{dt} = \underbrace{\mathfrak{T}_{\text{harvest}}(r_t)}_{\text{Income}} - \underbrace{\dot{\mathcal{M}}(t)}_{\text{Metabolic Cost}} - \underbrace{\gamma_{\text{leak}} B(t)}_{\text{Passive Dissipation}}

$$

where:
- $\mathfrak{T}_{\text{harvest}}(r_t)$ is the transduced energy from rewards (Definition {prf:ref}`def-metabolic-transducer`)
- $\dot{\mathcal{M}}(t)$ is the metabolic cost from Theorem {prf:ref}`thm-generalized-landauer-bound`
- $\gamma_{\text{leak}} \geq 0$ is the passive self-discharge rate (basal metabolic rate)

*Terminal Condition:* If $B(t) \leq 0$, the agent undergoes **Thermodynamic Death**. The metric collapses (Theorem {prf:ref}`thm-fading-metric-law`), inference halts, and the agent can no longer perform coherent computation.

:::

:::{div} feynman-prose
This equation is just accounting, but read it carefully:

$$
\frac{dB}{dt} = \text{(harvest)} - \text{(thinking costs)} - \text{(just existing costs)}

$$

The first term is income: what you extract from the environment by finding and exploiting resources. The second term is the Landauer cost of inference—every time you update your beliefs, you pay. The third term is the basal metabolic rate—the cost of just staying organized, keeping your proteins folded, maintaining the machinery.

That third term, $\gamma_{\text{leak}} B(t)$, is particularly insidious. Even if you do nothing—no thinking, no acting—you are still losing energy. The Second Law is patient, and it never sleeps. Organization decays. Batteries self-discharge. Living systems require continuous energy input just to maintain their structure.

This is why agents cannot simply "wait out" a bad situation indefinitely. The clock is always ticking.
:::

:::{admonition} The Three Energy Flows
:class: feynman-added tip

It helps to visualize the three flows separately:

**Income (Harvest):** $\mathfrak{T}_{\text{harvest}}(r_t)$
- Requires finding resources (positive reward)
- Bounded by transduction efficiency $\eta$
- Zero when $r_t \leq 0$

**Active Expenditure (Metabolism):** $\dot{\mathcal{M}}(t)$
- Proportional to how hard you are thinking
- Higher when sharpening beliefs rapidly
- Can be reduced by "System 1" operation

**Passive Drain (Leak):** $\gamma_{\text{leak}} B(t)$
- Proportional to current reserves
- Cannot be reduced to zero
- The inescapable tax of existence
:::

:::{prf:theorem} The Autopoietic Inequality
:label: thm-autopoietic-inequality

Let $\tau > 0$ be a target survival horizon. A **sufficient condition** for the agent to survive at time $\tau$ (i.e., $B(\tau) > 0$) is:

$$
\int_0^\tau \left( \mathfrak{T}_{\text{harvest}}(r_t) - \dot{\mathcal{M}}(t) \right) dt > \gamma_{\text{leak}} \int_0^\tau B(t) \, dt - B_0

$$

*Equivalently:* The time-averaged **Net Harvest Rate** must be positive:

$$
\langle \mathfrak{T} - \dot{\mathcal{M}} \rangle_\tau > \gamma_{\text{leak}} \langle B \rangle_\tau - \frac{B_0}{\tau}

$$

*Proof.*
Integrate the battery ODE (Axiom {prf:ref}`ax-energy-conservation-battery`):

$$
B(\tau) - B_0 = \int_0^\tau \mathfrak{T}(r_t) \, dt - \int_0^\tau \dot{\mathcal{M}}(t) \, dt - \gamma_{\text{leak}} \int_0^\tau B(t) \, dt

$$

Requiring $B(\tau) > 0$ and rearranging yields the inequality. $\square$

*Physical interpretation:* The agent must harvest more energy than it dissipates. This is the **autopoietic closure condition**—the system must actively maintain its own organization against thermodynamic decay.

:::

:::{div} feynman-prose
The Autopoietic Inequality is the survival condition. It says, in essence: **you must earn more than you spend**.

The word "autopoiesis" comes from the Greek for "self-making." It was coined by Maturana and Varela to describe systems that continuously regenerate themselves—living systems, essentially. The key insight is that living things are not just complicated machines; they are machines that must actively maintain their own existence. A rock can just sit there. A bacterium cannot—it must constantly do work to stay organized, to keep its membrane intact, to prevent its proteins from denaturing.

The inequality tells us the minimum performance an agent must achieve to survive. And notice that it is not just about maximizing reward—it is about maintaining a positive energy balance. An agent that pursues high-cost strategies (expensive inference) must also achieve high harvests. An agent with limited harvesting ability must economize on thinking.

This is the thermodynamic foundation of bounded rationality.
:::

:::{prf:corollary} The Survival Objective
:label: cor-survival-objective

The agent's fundamental objective is not reward maximization but **energy surplus maximization**:

$$
\mathcal{J}_{\text{survival}} = \mathbb{E}\left[ \int_0^\infty \left( \mathfrak{T}_{\text{harvest}}(r_t) - \dot{\mathcal{M}}(t) \right) e^{-\gamma_{\text{leak}} t} \, dt \right]

$$

Standard reward maximization $\max \mathbb{E}[\sum_t \gamma^t r_t]$ emerges as a degenerate case when:
1. Metabolic cost $\dot{\mathcal{M}} \to 0$ (free computation)
2. Transduction efficiency $\eta \to 1$ (perfect conversion)
3. Battery capacity $B_{\max} \to \infty$ (unlimited storage)

:::

:::{div} feynman-prose
Here is the punchline: **standard reinforcement learning is a limiting case**.

When we write down the usual RL objective—maximize expected discounted reward—we are implicitly assuming that thinking is free, energy conversion is perfect, and storage is unlimited. These are the assumptions of an idealized agent with infinite resources.

Real agents, whether biological or artificial, face constraints. The survival objective captures what they are *actually* optimizing: not raw reward, but net energy surplus. The discount factor $\gamma$ in standard RL corresponds to the leak rate $\gamma_{\text{leak}}$—it is not arbitrary but reflects the thermodynamic reality that future energy is worth less than present energy because you have to survive to get there.

This perspective resolves a lot of puzzles in RL. Why do agents often seem "risk-averse" in ways that pure expected-value maximizers should not be? Because they are not maximizing expected value—they are maximizing survival. Why do biological organisms interrupt task pursuit to seek food? Because the survival objective includes the metabolic cost of continued operation.
:::

:::{admonition} Connection to RL #34: Reward Maximization as Infinite-Battery Limit
:class: note
:name: conn-rl-34
**The General Law (Fragile Agent):** Survival objective $\mathcal{J}_{\text{survival}} = \mathbb{E}[\int (\mathfrak{T}(r) - \dot{\mathcal{M}}) \, dt]$

**The Degenerate Limit:** $B \to \infty$ (inexhaustible battery), $\dot{\mathcal{M}} \to 0$ (free computation)

**The Special Case (Standard RL):** $\max \mathbb{E}[\sum_t \gamma^t r_t]$ (pure reward maximization)

**What the generalization offers:**
- Explains *why* agents must be computationally efficient
- Derives intrinsic motivation for energy-seeking behavior
- Provides termination criterion (death) without external specification
- Connects RL objective to thermodynamic first principles
:::



(sec-the-fading-metric-energy-dependent-geometry)=
## The Fading Metric: Energy-Dependent Geometry

The battery $B(t)$ is not merely a scalar reward modifier (conservative case)—it is a **constraint on the geometry
itself**. Without energy, the agent cannot maintain the precise neural representations required for a high-resolution
metric ({ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits`). We derive this from
Fisher Information principles.

:::{div} feynman-prose
Now we arrive at what I consider the most striking result in this chapter. We have been treating the latent geometry—the metric tensor $G$—as if it were a fixed property of the agent's architecture. But it is not fixed. It costs energy to maintain.

Think about what the metric means operationally. It measures distinguishability: how well the agent can tell nearby states apart. High metric resolution means fine discrimination—the agent can distinguish states that are close together in latent space. But distinguishing things requires precision, and precision requires energy.

When your blood sugar drops, your thinking gets fuzzy. When a computer runs low on power, calculations become unreliable. This is not a metaphor—it is a direct consequence of the thermodynamics of information. Maintaining sharp probability distributions costs energy. When energy runs low, the distributions blur, the metric fades, and distinct concepts become indistinguishable.

This is the Fading Metric Law.
:::

:::{prf:theorem} The Information-Maintenance Cost
:label: thm-information-maintenance-cost

Maintaining Fisher Information $I_F$ on the latent manifold $(\mathcal{Z}, G)$ requires continuous energy expenditure:

$$
\dot{E}_{\text{maintain}} \geq \frac{1}{2} T_c \cdot I_F

$$

where $T_c$ is the cognitive temperature ({prf:ref}`def-cognitive-temperature`) and $I_F$ is the Fisher Information of the belief distribution.

*Proof sketch.*
1. **Fisher Information definition:** For belief density $\rho(z)$ on $(\mathcal{Z}, G)$:
   $$I_F = \mathbb{E}_\rho\left[ \|\nabla \ln \rho\|_G^2 \right] = \int_\mathcal{Z} \rho(z) \|\nabla \ln \rho(z)\|_{G^{-1}}^2 \, d\mu_G(z)$$

2. **de Bruijn identity** {cite}`stam1959some,cover2006elements`: Under diffusion $d\rho/dt = T_c \Delta_G \rho$, entropy evolves as:
   $$\frac{dH[\rho]}{dt} = \frac{1}{2} I_F[\rho]$$
   Entropy increases at rate proportional to Fisher Information.

3. **Landauer cost:** By Theorem {prf:ref}`thm-generalized-landauer-bound`, maintaining entropy against diffusion requires:
   $$\dot{E}_{\text{maintain}} \geq T_c \left| \frac{dH}{dt} \right| = \frac{1}{2} T_c \cdot I_F$$

4. **Interpretation:** Sharp probability distributions (high $I_F$) cost more to maintain. $\square$

:::

:::{div} feynman-prose
The de Bruijn identity is one of those beautiful results in information theory that does not get the attention it deserves. It says that under thermal noise, entropy increases at a rate proportional to Fisher Information.

Why is that? Well, Fisher Information measures how "peaked" your distribution is—how much the log-probability varies as you move around. A sharply peaked distribution has high Fisher Information. And a sharply peaked distribution is exactly the kind that diffusion attacks most effectively. Thermal noise spreads things out, and the sharper your peak, the faster it spreads.

To maintain a sharp distribution against diffusion—to keep your beliefs precise—you have to continuously pump entropy out of the system. That takes energy. The Landauer bound tells you how much: at least $T_c$ per nat of entropy removed.

So high-resolution representations require continuous energy expenditure just to maintain. No energy, no resolution.
:::

:::{prf:theorem} The Fading Metric Law
:label: thm-fading-metric-law

When available energy $B(t)$ falls below the maintenance requirement, the effective metric contracts. The **effective metric** is:

$$
G_{ij}^{\text{eff}}(z, B) = f\left(\frac{B}{B_{\text{crit}}}\right) \cdot G_{ij}(z)

$$

where:
- $G_{ij}(z)$ is the full-capacity metric (Theorem {prf:ref}`thm-capacity-constrained-metric-law`)
- $B_{\text{crit}}$ is the **critical energy** required to sustain full metric resolution
- $f: [0, \infty) \to [0, 1]$ is the **fading function** with $f(0) = 0$, $\lim_{x \to \infty} f(x) = 1$

**Specific form:** The fading function satisfying thermodynamic constraints is:

$$
f(x) = 1 - e^{-x}

$$

This gives exponential saturation: $f(x) \approx x$ for $x \ll 1$ (linear regime) and $f(x) \approx 1$ for $x \gg 1$ (saturation).

*Proof sketch.*
1. **Fisher metric interpretation:** The metric $G$ encodes distinguishability—the statistical distance between nearby states. Formally, $G_{ij} = \mathbb{E}[\partial_i \ln p \cdot \partial_j \ln p]$ where $p$ is the encoding distribution.

2. **Signal-to-noise scaling:** Neural signals have SNR proportional to available energy:
   $$\text{SNR} \propto \sqrt{\frac{E_{\text{available}}}{E_{\text{noise}}}} = \sqrt{\frac{B}{B_{\text{crit}}}}$$

3. **Fisher Information scaling:** Since Fisher Information scales as SNR²:
   $$I_F^{\text{eff}} \propto \text{SNR}^2 \propto \frac{B}{B_{\text{crit}}}$$

4. **Metric scaling:** The metric tensor scales with Fisher Information:
   $$G^{\text{eff}} \propto I_F^{\text{eff}} \propto \frac{B}{B_{\text{crit}}} \quad \text{for } B \ll B_{\text{crit}}$$

5. **Saturation:** For $B \gg B_{\text{crit}}$, the metric saturates at $G$ (maximum resolution). The exponential form $f(x) = 1 - e^{-x}$ interpolates smoothly between these regimes. $\square$

:::

:::{div} feynman-prose
Let me draw the picture that should be in your head.

When you have plenty of energy, the metric is sharp. Points in latent space that represent different concepts are far apart—easily distinguishable. The landscape of your mind is high-resolution.

Now start draining the battery. The metric contracts. Those distinct concepts? They are getting closer together. The peaks in your probability distribution are spreading out, blurring. What used to be two clearly separated ideas are now becoming one fuzzy blob.

At zero energy, the metric collapses to zero. Everything is the same. There is no distinguishability left. This is the geometric manifestation of death: not an explosion or a shutdown, but a collapse of distinctions. The agent can no longer tell different things apart, so it can no longer act coherently.

The fading function $f(x) = 1 - e^{-x}$ captures the physics nicely:
- Near zero, it is linear: $f(x) \approx x$. Every bit of energy helps, proportionally.
- As $x$ gets large, it saturates. Once you have "enough" energy, more does not help much—you have already reached the resolution limit of your architecture.

The critical energy $B_{\text{crit}}$ is where you are at about 63% of full resolution. Above that, you are mostly fine. Below that, things degrade fast.
:::

:::{admonition} Visualizing the Fading Metric
:class: feynman-added note

Imagine a 2D latent space with a Gaussian bump representing a concept:

**High Energy ($B \gg B_{\text{crit}}$):**
- Sharp, narrow bump
- Large distances between different concepts
- Clear distinctions possible

**Critical Energy ($B \approx B_{\text{crit}}$):**
- Bump spreading out
- Distances shrinking
- Distinctions becoming difficult

**Low Energy ($B \ll B_{\text{crit}}$):**
- Flat, spread-out distribution
- Distances nearly zero
- Everything looks the same

This is not just an analogy—it is the literal geometric description of what happens to your representational capacity as energy depletes.
:::

:::{prf:corollary} Consequences of Metric Fading
:label: cor-metric-fading-consequences

As $B(t) \to 0$, the following degenerations occur:

1. **Resolution Loss:** Geodesic distances collapse:
   $$d_G^{\text{eff}}(z, z') = \sqrt{f(B/B_{\text{crit}})} \cdot d_G(z, z') \to 0$$
   Distinct concepts become indistinguishable.

2. **Inertia Loss:** The mass term in the geodesic SDE (Definition {prf:ref}`def-bulk-drift-continuous-flow`) vanishes. The agent loses momentum and becomes dominated by thermal noise.

3. **Causal Dissolution:** The Causal Information Bound ({ref}`sec-causal-information-bound`, Theorem {prf:ref}`thm-causal-information-bound`) collapses:
   $$I_{\max}^{\text{eff}} = \frac{\text{Area}(\partial\mathcal{Z})}{4\ell_L^2} \cdot f(B/B_{\text{crit}}) \to 0$$
   The agent's representational capacity vanishes.

4. **Control Loss:** The policy gradient $\nabla_z \Phi_{\text{eff}}$ scales with metric, so control authority degrades.

:::

:::{prf:corollary} The Starvation-Hallucination Regime
:label: cor-starvation-hallucination

As $B(t) \to 0$, the signal-to-noise ratio of internal dynamics degrades:

$$
\text{SNR}_{\text{dynamics}} = \frac{\|v\|_{G^{\text{eff}}}^2}{2T_c} \propto f(B/B_{\text{crit}}) \to 0

$$

In this regime:
- The drift term $v = -G^{-1} \nabla \Phi$ vanishes relative to diffusion $\sqrt{2T_c} dW$
- The agent performs a **random walk** in latent space
- Internal trajectories are indistinguishable from noise: **hallucination**

*Biological analogue:* Hypoglycemia causes confusion, disorientation, and hallucinations before coma—the same phenomenology predicted by metric fading. See also the Cognitive Temperature (Definition {prf:ref}`def-cognitive-temperature`) which controls the noise-to-signal ratio in latent dynamics.

:::

:::{div} feynman-prose
The Starvation-Hallucination Regime is perhaps the eeriest prediction of this theory.

What happens when you are starving? When your blood sugar drops dangerously low? You do not just slow down—you start to hallucinate. You see things that are not there. Your thinking becomes disorganized. You lose the ability to distinguish reality from fantasy.

The Fading Metric Law explains why. As energy depletes:
1. The metric contracts, reducing your ability to distinguish states
2. The signal (purposeful drift toward goals) shrinks
3. The noise (thermal fluctuations) stays constant
4. Eventually, noise dominates signal

In this regime, your internal state is just diffusing randomly through latent space. The trajectory is no longer guided by goals or beliefs—it is pure Brownian motion. And what does random motion through a representational space look like from the inside?

Hallucination.

The connection between starvation and hallucination is not a bug—it is a thermodynamic consequence of trying to run a cognitive system without fuel. The geometry of thought literally dissolves.
:::

::::{admonition} Physics Isomorphism: Johnson-Nyquist Noise
:class: note
:name: pi-johnson-nyquist

**In Physics:** Johnson-Nyquist noise {cite}`johnson1928thermal,nyquist1928thermal` in resistors has spectral density:
$$S_V(f) = 4 k_B T R$$
where $R$ is resistance and $T$ is temperature. The SNR of any electrical signal is limited by this thermal noise floor.

**In Implementation:** Neural representations are subject to analogous noise. When the "power supply" (battery $B$) is low:
- Signal amplitude decreases (less energy for spike generation)
- Noise floor remains constant (thermal/synaptic noise)
- SNR degrades as $\sqrt{B}$
- Fisher Information (metric) degrades as $B$

**Correspondence:**

| Physics | Agent |
|:--------|:------|
| Thermal noise $4k_B T R$ | Synaptic/neural noise |
| Signal power | Battery $B(t)$ |
| SNR $\propto P/N$ | Metric scaling $f(B)$ |
| Signal degradation | Hallucination |

::::



(sec-homeostatic-control-battery-potential)=
## Homeostatic Control: The Battery Potential

How does the agent "know" to seek energy when depleted? We introduce a **Homeostatic Potential** that modifies the value landscape based on battery state.

:::{div} feynman-prose
We have established that the agent will die if its battery runs out. But how does the agent know to do something about it? Where does the drive to seek energy come from?

In standard RL, you have to hand-design a reward signal that says "eating is good." But that seems backwards. An agent that does not eat dies. The preference for eating should emerge from the physics, not be imposed from outside.

And indeed it does. The trick is that battery state modifies the value landscape. When you are well-fed, the "food" region of latent space is mildly attractive. When you are starving, it becomes overwhelmingly attractive—so attractive that it dominates all other considerations.

This is homeostatic control, and it emerges naturally from the autopoietic structure.
:::

:::{prf:definition} The Homeostatic Potential
:label: def-homeostatic-potential

The battery level $B(t)$ induces a scalar potential field acting on the policy:

$$
\Phi_{\text{homeo}}(z, B) = \frac{\lambda_{\text{surv}}}{B + \epsilon} \cdot \mathbb{1}[z \in \mathcal{Z}_{\text{food}}]

$$

where:
- $\lambda_{\text{surv}} > 0$ is the **survival weight** (dimensionless priority)
- $\epsilon > 0$ is a regularization constant preventing singularity
- $\mathcal{Z}_{\text{food}} \subset \mathcal{Z}$ is the **food region** (states where $\mathfrak{T}(r) > 0$)

*Units:* $[\Phi_{\text{homeo}}] = [\Phi_{\text{task}}] = \text{nats}$ (log-probability scale).

:::

:::{div} feynman-prose
The form of this potential is worth thinking about. It is inversely proportional to battery level: $\Phi_{\text{homeo}} \propto 1/B$.

When $B$ is large (well-fed), the homeostatic potential is small. The agent can focus on other things—tasks, exploration, whatever it was doing.

When $B$ is small (starving), the homeostatic potential explodes. It becomes $1/\epsilon$ as $B \to 0$, which is very large. This creates an enormous gradient pointing toward the food region.

The agent does not need to "know" it is hungry in any reflective sense. The physics does the work. Low battery creates a potential well so deep that the agent cannot help but fall toward food. This is the mechanical implementation of hunger.
:::

:::{prf:theorem} The Augmented Value Equation
:label: thm-augmented-value-equation

The total effective potential combines task and homeostatic contributions:

$$
\Phi_{\text{total}}(z, B) = \Phi_{\text{task}}(z) + \Phi_{\text{homeo}}(z, B)

$$

The value function satisfies the augmented screened Poisson equation ({ref}`sec-the-reward-field-value-forms-and-hodge-geometry`):

$$
(-\Delta_{G^{\text{eff}}} + \kappa^2) V = \rho_r + \rho_{\text{homeo}}

$$

where:
- $G^{\text{eff}} = f(B/B_{\text{crit}}) \cdot G$ is the faded metric (Theorem {prf:ref}`thm-fading-metric-law`)
- $\rho_{\text{homeo}} = -\Delta \Phi_{\text{homeo}}$ is the homeostatic source term
- The screening mass $\kappa = -\ln \gamma$ remains unchanged

*Consequence:* Both the metric (geometry) and the source term (drive) depend on battery state.

:::

:::{prf:corollary} Priority Inversion at Low Battery
:label: cor-priority-inversion

As $B \to 0$:

1. **Homeostatic dominance:** $\Phi_{\text{homeo}} \propto 1/B \to \infty$ while $\Phi_{\text{task}}$ remains bounded
2. **Gradient steering:** $\nabla_z \Phi_{\text{total}} \approx \nabla_z \Phi_{\text{homeo}}$ points toward $\mathcal{Z}_{\text{food}}$
3. **Priority inversion:** Task objectives become irrelevant; survival dominates

*Behavioral consequence:* A starving agent abandons task pursuit and seeks energy. This behavior emerges from the thermodynamic structure of autopoietic systems.

:::

:::{div} feynman-prose
Priority Inversion is a technical name for a familiar phenomenon: when you are really hungry, you cannot think about anything else.

Mathematically, it works like this. The total potential is the sum of task and homeostatic terms. The task potential is bounded—there is only so much value in any task. But the homeostatic potential goes like $1/B$, which is unbounded as $B \to 0$.

So there is always some battery level below which the homeostatic gradient dominates. Below that threshold, the agent will abandon whatever it was doing and seek energy. Not because it "decides" to—because the gradient field leaves it no choice.

This explains a lot of animal behavior. Why does a bird interrupt nest-building to forage? Why does a programmer get up from an intense debugging session to eat lunch? Not because they consciously calculate that food is more important (though they might tell that story). Because their internal gradient field has been overwhelmed by the homeostatic term.

The beautiful thing is that this behavior is derived, not designed. We did not put in a rule saying "seek food when hungry." The rule emerged from the thermodynamics of self-maintaining systems.
:::

:::{admonition} Connection to RL #35: Intrinsic Motivation as Battery-Independent Limit
:class: note
:name: conn-rl-35
**The General Law (Fragile Agent):** Battery-modulated homeostatic potential $\Phi_{\text{homeo}}(z, B) = \lambda_{\text{surv}}/(B + \epsilon)$

**The Degenerate Limit:** $B \to \infty$ or $\lambda_{\text{surv}} \to 0$

**The Special Case (Standard RL):** Pure task reward / curiosity-driven exploration without survival pressure

**What the generalization offers:**
- Derives priority of survival over task completion
- Explains why biological agents interrupt tasks for food-seeking
- Provides principled interpolation between exploration and exploitation based on energy state
- Grounds "intrinsic motivation" in thermodynamic necessity
:::



(sec-thermal-management-carnot-bound)=
## Thermal Management and the Carnot Bound

The transduction efficiency $\eta$ is not a free parameter—it is bounded by thermodynamics. We derive this bound and its consequences.

:::{div} feynman-prose
We have been treating the transduction efficiency $\eta$ as a number between 0 and 1. But what determines its actual value? Is it just engineering—make a better transducer, get a higher $\eta$?

No. There is a fundamental limit. The Metabolic Transducer is a heat engine—it extracts work by exploiting a temperature difference between the agent and the environment. And heat engines are constrained by the Carnot limit, the most fundamental bound in thermodynamics.

This has a striking consequence: the agent must maintain itself at a lower temperature than its environment. If it heats up to environmental temperature, the engine stops working, and harvesting becomes impossible.
:::

:::{prf:theorem} The Carnot Bound on Transduction
:label: thm-carnot-transduction-bound

The transduction efficiency is bounded by the Carnot limit:

$$
\eta \leq \eta_{\text{Carnot}} = 1 - \frac{T_c}{T_{\text{env}}}

$$

where $T_c$ is the agent's cognitive temperature and $T_{\text{env}}$ is the environmental temperature.

*Proof.* By the Second Law of Thermodynamics, no heat engine can exceed Carnot efficiency when operating between reservoirs at temperatures $T_{\text{hot}} = T_{\text{env}}$ and $T_{\text{cold}} = T_c$. The Metabolic Transducer is such an engine—it extracts work from the temperature differential between environment and internal state. $\square$

*Consequence:* The agent must maintain $T_c < T_{\text{env}}$ (a thermal gradient) to extract any work. If $T_c \geq T_{\text{env}}$, then $\eta \leq 0$ and no harvesting is possible.

:::

:::{div} feynman-prose
Here is Carnot's argument in a nutshell. You want to extract work from heat. You have a hot reservoir (the environment) and a cold reservoir (the agent's innards). Heat flows from hot to cold—that is the Second Law. As it flows, some of it can be converted to work, but not all.

The maximum efficiency is:

$$
\eta_{\text{Carnot}} = 1 - \frac{T_{\text{cold}}}{T_{\text{hot}}} = 1 - \frac{T_c}{T_{\text{env}}}

$$

This is a hard limit. No cleverness, no engineering trick can exceed it. It is written into the laws of physics.

For the agent, this means that cognitive temperature $T_c$ is not just a parameter controlling exploration—it is a constraint on survivability. A very hot agent (high $T_c$) might be very exploratory, but it will have terrible transduction efficiency. A very cold agent will have great efficiency but might be too deterministic.

The sweet spot is somewhere in between, and finding it is part of what the Universal Governor does.
:::

:::{prf:definition} The Waste Heat Flux
:label: def-waste-heat-flux

The **Waste Heat Flux** is the rate at which the agent must dump entropy to the environment:

$$
\dot{Q}_{\text{waste}} = (1 - \eta) \cdot \mathfrak{T}_{\text{gross}}(r_t) + \dot{\mathcal{M}}(t)

$$

where $\mathfrak{T}_{\text{gross}} = k_B T_{\text{env}} \cdot \mathcal{I}_{\text{util}}(r_t)$ is the gross transduction before efficiency losses.

*Units:* $[\dot{Q}_{\text{waste}}] = \text{Watts}$ (power).

*Interpretation:* All non-useful energy becomes waste heat that must be radiated to maintain thermal equilibrium.

:::

:::{prf:corollary} The Thermal Runaway Condition
:label: cor-thermal-runaway

Let $\dot{Q}_{\text{radiate}}$ be the maximum heat dissipation rate (determined by surface area, environment, cooling mechanisms). If:

$$
\dot{Q}_{\text{waste}} > \dot{Q}_{\text{radiate}}

$$

then the agent's internal temperature $T_c$ increases. This triggers a positive feedback loop:

1. $T_c \uparrow$ $\Rightarrow$ $\eta_{\text{Carnot}} = 1 - T_c/T_{\text{env}} \downarrow$
2. Lower $\eta$ $\Rightarrow$ more waste heat for same harvesting
3. More waste heat $\Rightarrow$ $T_c \uparrow$ (feedback)

*Terminal state:* $T_c \to T_{\text{env}}$, $\eta \to 0$, no harvesting possible, death by thermal runaway.

*Biological analogue:* Hyperthermia/heat stroke—metabolic rate increases with temperature, but cooling capacity is bounded, leading to runaway heating.

:::

:::{div} feynman-prose
Thermal runaway is a nasty way to die. Let me trace through the feedback loop.

Suppose the agent is harvesting energetically and thinking hard. Both activities generate waste heat. If the cooling system cannot keep up, the agent heats up a little. But now $T_c$ is higher, so Carnot efficiency is lower. With lower efficiency, more of the harvested energy becomes waste heat. Which heats up the agent more. Which lowers efficiency further...

You see where this is going. It is a positive feedback loop, and positive feedback loops tend to run away to extremes. In this case, the extreme is $T_c = T_{\text{env}}$, at which point $\eta = 0$ and the agent can no longer harvest at all. Game over.

Heat stroke works exactly this way. Your body generates heat from metabolism. Normally, sweating and blood flow to the skin dump this heat to the environment. But if the environment is too hot, or you are exercising too hard, or your cooling system fails, you start heating up. And metabolic rate increases with temperature (chemistry goes faster when it is hot), so you generate even more heat. The feedback can become lethal very quickly.

For artificial agents, thermal management is equally important. GPUs throttle when they get hot. Data centers spend enormous resources on cooling. The Carnot bound is not just theoretical—it shapes the design of every computational system.
:::

:::{admonition} Thermal Death vs. Starvation Death
:class: feynman-added warning

There are two ways to die in this framework:

**Starvation Death (Battery Depletion):**
- $B(t) \to 0$
- Metric fades, inference degrades, hallucination
- Gradual onset, potentially recoverable if food found
- Final state: frozen, non-functional

**Thermal Death (Runaway):**
- $T_c \to T_{\text{env}}$
- Efficiency collapses, waste heat dominates
- Rapid onset once triggered, hard to reverse
- Final state: equilibrated, non-functional

Both are **irreversible** within an episode. The agent should monitor both and take preemptive action (Node 67, 69).
:::

:::{prf:definition} The Thermal Operating Envelope
:label: def-thermal-operating-envelope

The agent is **thermally viable** if there exists a steady-state solution to:

$$
\dot{Q}_{\text{waste}}(T_c) = \dot{Q}_{\text{radiate}}(T_c)

$$

with $T_c < T_{\text{env}}$ and $\eta(T_c) > \eta_{\min}$ where $\eta_{\min}$ is the minimum efficiency for survival (from Theorem {prf:ref}`thm-autopoietic-inequality`).

The **Thermal Operating Envelope** is the region in $(T_c, \dot{\mathcal{M}}, \dot{Q}_{\text{radiate}})$ space where this condition holds.

:::



(sec-implementation-metabolic-battery)=
## Implementation: The MetabolicBattery Module

We provide the reference implementation linking the Sieve, the Governor, and the Reward signal.

:::{div} feynman-prose
Now let us make this concrete with code. The following implementation captures the key concepts: the transducer converting reward to energy, the battery dynamics, the fading metric, and the diagnostic nodes.

Pay attention to how the pieces fit together. The `transducer` method implements the Szilard correspondence—it takes reward in nats and outputs energy in joules. The `get_metric_scaling` method implements the fading function. The `update` method implements the full battery dynamics from Axiom {prf:ref}`ax-energy-conservation-battery`.
:::

```python
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class MetabolismConfig:
    """Configuration for the Metabolic Transducer and Battery.

    All parameters have physical units and interpretations from {ref}`sec-the-metabolic-transducer-autopoiesis-and-the-szilard-engine`.
    """
    battery_capacity: float = 100.0      # B_max: Maximum stored energy (Joules)
    initial_battery: float = 50.0        # B_0: Initial endowment
    base_leak_rate: float = 0.01         # gamma_leak: Passive dissipation rate (1/step)
    joules_per_nat: float = 1.0          # k_B * T_env: Boltzmann conversion factor
    carnot_efficiency: float = 0.5       # eta: Transduction efficiency
    critical_threshold: float = 5.0      # B_crit: Threshold for metric fading
    env_temperature: float = 300.0       # T_env: Environmental temperature (K)
    survival_weight: float = 10.0        # lambda_surv: Homeostatic priority


class MetabolicBattery(nn.Module):
    """
    Implements the Metabolic Transducer (Definition 36.1.4) and Internal Battery
    (Definition 36.2.1). Modulates the metric G based on energy availability
    (Theorem 36.3.2).

    References:
        - Transducer: Definition `def-metabolic-transducer`
        - Battery dynamics: Axiom `ax-energy-conservation-battery`
        - Fading Metric: Theorem `thm-fading-metric-law`
        - Autopoietic Inequality: Theorem `thm-autopoietic-inequality`
    """

    def __init__(self, config: MetabolismConfig):
        super().__init__()
        self.config = config
        self.register_buffer('battery', torch.tensor(config.initial_battery))
        self.register_buffer('cognitive_temp', torch.tensor(config.env_temperature * 0.8))
        self.register_buffer('is_dead', torch.tensor(False))

        # Running statistics for diagnostic nodes
        self.register_buffer('harvest_ema', torch.tensor(0.0))
        self.register_buffer('cost_ema', torch.tensor(0.0))
        self.ema_decay = 0.99

    def transducer(self, reward_nats: torch.Tensor) -> torch.Tensor:
        """
        Metabolic Transducer operator T_harvest (Definition 36.1.4).

        Args:
            reward_nats: Reward signal r_t in nats

        Returns:
            Energy flux E_in in Joules/step
        """
        eta = self.get_carnot_efficiency()
        # Only positive rewards harvest energy (can't extract work from bad states)
        positive_reward = torch.clamp(reward_nats, min=0.0)
        return eta * self.config.joules_per_nat * positive_reward

    def get_carnot_efficiency(self) -> torch.Tensor:
        """
        Compute current Carnot efficiency (Theorem 36.5.1).

        Returns:
            eta = 1 - T_c / T_env, clamped to [0, 1]
        """
        eta = 1.0 - self.cognitive_temp / self.config.env_temperature
        return torch.clamp(eta, 0.0, self.config.carnot_efficiency)

    def update(self, reward_nats: float, metabolic_cost_joules: float) -> Tuple[float, dict]:
        """
        Execute one step of the thermodynamic loop (Axiom 36.2.2).

        Args:
            reward_nats: Reward signal r_t
            metabolic_cost_joules: Metabolic cost M_dot from {ref}`sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics`

        Returns:
            delta: Net energy change
            diagnostics: Dictionary of diagnostic values for Nodes 67-70
        """
        if self.is_dead:
            return 0.0, {'alive': False, 'battery': 0.0}

        # 1. Transducer: Harvest energy from reward
        energy_in = self.transducer(torch.tensor(reward_nats))

        # 2. Metabolic cost (Landauer dissipation)
        energy_out = torch.tensor(metabolic_cost_joules)

        # 3. Passive leak (basal metabolic rate)
        leak = self.config.base_leak_rate * self.battery

        # 4. Battery dynamics (Axiom 36.2.2)
        delta = energy_in - energy_out - leak
        new_battery = torch.clamp(
            self.battery + delta,
            0.0,
            self.config.battery_capacity
        )
        self.battery.copy_(new_battery)

        # 5. Update EMAs for diagnostic nodes
        self.harvest_ema.copy_(
            self.ema_decay * self.harvest_ema + (1 - self.ema_decay) * energy_in
        )
        self.cost_ema.copy_(
            self.ema_decay * self.cost_ema + (1 - self.ema_decay) * energy_out
        )

        # 6. Check termination (Node 67: AutopoiesisCheck)
        if self.battery <= 0:
            self.is_dead.copy_(torch.tensor(True))

        # 7. Compute diagnostics
        diagnostics = {
            'alive': not self.is_dead.item(),
            'battery': self.battery.item(),
            'metric_scaling': self.get_metric_scaling(),
            'harvest_efficiency': self.get_harvest_efficiency(),
            'thermal_margin': self.get_thermal_margin(),
        }

        return delta.item(), diagnostics

    def get_metric_scaling(self) -> float:
        """
        Fading Metric scaling factor f(B/B_crit) (Theorem 36.3.2).

        Returns:
            f in [0, 1]: Multiply latent metric G by this factor
        """
        if self.is_dead:
            return 0.0
        x = self.battery / self.config.critical_threshold
        return (1.0 - torch.exp(-x)).item()

    def get_homeostatic_drive(self) -> float:
        """
        Homeostatic potential strength (Definition 36.4.1).

        Returns:
            Phi_homeo = lambda_surv / (B + epsilon)
        """
        return self.config.survival_weight / (self.battery.item() + 1e-3)

    def get_harvest_efficiency(self) -> float:
        """
        Node 68: Harvest efficiency ratio <T(r)> / <M_dot>.

        Returns:
            Ratio > 1 means sustainable, < 1 means dying
        """
        if self.cost_ema < 1e-6:
            return float('inf')
        return (self.harvest_ema / self.cost_ema).item()

    def get_thermal_margin(self) -> float:
        """
        Node 69: Thermal safety margin T_env - T_c.

        Returns:
            Positive = safe, zero/negative = thermal runaway
        """
        return (self.config.env_temperature - self.cognitive_temp).item()

    def check_thermal_runaway(self, waste_heat: float, max_dissipation: float) -> bool:
        """
        Check thermal runaway condition (Corollary 36.5.3).

        Args:
            waste_heat: Current Q_dot_waste
            max_dissipation: Maximum Q_dot_radiate

        Returns:
            True if thermal runaway is occurring
        """
        return waste_heat > max_dissipation
```

:::{div} feynman-prose
A few things to notice in the code:

1. **The transducer clamps negative rewards to zero.** You cannot extract work from being in a bad place—that is not how thermodynamics works. Negative reward means you are in a high-entropy configuration with nothing to harvest.

2. **The fading function is `1 - exp(-x)`.** This is the specific form from Theorem {prf:ref}`thm-fading-metric-law`. You can see how it gives linear behavior near zero and saturates to 1 for large arguments.

3. **The diagnostic methods compute quantities for Nodes 67-70.** These are the autopoiesis monitors that tell us whether the agent is on track to survive.

4. **The EMAs smooth out the harvest and cost signals.** Instantaneous values are noisy; the exponential moving average gives a more reliable estimate of the agent's metabolic trajectory.
:::



(sec-diagnostic-nodes-autopoiesis)=
## Diagnostic Nodes 67-70: Autopoiesis

We add four diagnostic nodes to the Sieve monitoring autopoietic viability.

:::{div} feynman-prose
These four nodes are the vital signs monitors of the autopoietic system. They tell you whether the agent is alive, whether it is sustainable, whether it is overheating, and whether its representational capacity is intact.

In a hospital, you monitor heart rate, blood pressure, temperature, and oxygen saturation. For an autopoietic agent, these four nodes play the analogous role.
:::

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **67** | **AutopoiesisCheck** | Battery | Survival | Is the agent alive? | $B(t) > 0$ | $O(1)$ |
| **68** | **HarvestEfficiencyCheck** | Transducer | Sustainability | Is lifestyle sustainable? | $\langle\mathfrak{T}\rangle / \langle\dot{\mathcal{M}}\rangle > 1$ | $O(1)$ |
| **69** | **ThermalRunawayCheck** | Cooling | Stability | Is thermal equilibrium stable? | $T_c < T_{\text{env}}$ | $O(1)$ |
| **70** | **MetricFadingCheck** | Geometry | Degradation | Is metric resolution adequate? | $f(B/B_{\text{crit}}) > \epsilon_{\text{fade}}$ | $O(1)$ |



(node-67)=
**Node 67: AutopoiesisCheck**

*Trigger condition:* $B(t) \leq 0$

*Interpretation:* The agent has exhausted its energy reserves. This is **Thermodynamic Death**—an irreversible terminal state.

*Remediation:* None (death is irreversible in the current episode). In meta-learning or population settings:
- Initialize next generation with higher $\eta$ or more conservative policy
- Select for lineages with positive $\langle \mathfrak{T} - \dot{\mathcal{M}} \rangle$

:::{div} feynman-prose
Node 67 is the final check—are you still alive? If the battery hits zero, it is game over. No remediation is possible because there is no energy left to do anything with.

The interesting question is what happens in populations or across training runs. Evolution selects for survival, and agents that die contribute nothing to the next generation. Over time, this pressure should shape policies toward sustainable energy balance.
:::



(node-68)=
**Node 68: HarvestEfficiencyCheck**

*Trigger condition:* $\langle \mathfrak{T}(r) \rangle_T / \langle \dot{\mathcal{M}} \rangle_T < 1$ (time-averaged over window $T$)

*Interpretation:* The agent is burning more energy than it collects. It is on a trajectory toward death.

*Remediation:* Governor Intervention:
1. Reduce Cognitive Temperature $T_c$ (enter "hibernate" / System 1 mode; see fast/slow phase transition in {ref}`sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics`)
2. Suppress Curiosity coefficient $\beta_{\text{exp}}$ (stop exploring)
3. Increase Homeostatic Weight $\lambda_{\text{surv}}$ (focus on food-seeking)
4. Trigger Deliberation Stopping (Theorem {prf:ref}`thm-deliberation-optimality-condition`) earlier

:::{div} feynman-prose
Node 68 is the early warning system. If your harvest efficiency drops below 1, you are on a death spiral—it is just a matter of time before Node 67 triggers.

The remediation options are all forms of conservation. Reduce thinking (lower $T_c$). Stop exploring. Focus on survival. Cut your metabolic losses.

This is the thermodynamic basis of stress response. When resources are scarce, the agent should become more conservative, more focused, less exploratory. Not because someone programmed in that response, but because the physics demands it.
:::



(node-69-metabolic)=
**Node 69: ThermalRunawayCheck**

*Trigger condition:* $T_c \geq T_{\text{env}} - \delta_{\text{thermal}}$ (approaching thermal parity)

*Interpretation:* The agent is overheating. Carnot efficiency is collapsing.

*Remediation:*
1. Reduce metabolic rate $\dot{\mathcal{M}}$ (pause inference)
2. Increase cooling (if controllable)
3. Reduce transduction rate (take fewer actions)
4. Enter "sleep" mode: reflective boundary, zero action, maximize heat dissipation

:::{div} feynman-prose
Node 69 monitors for thermal runaway. If $T_c$ approaches $T_{\text{env}}$, the Carnot efficiency goes to zero and the positive feedback loop from Corollary {prf:ref}`cor-thermal-runaway` kicks in.

The remediation is to cool down—stop working so hard, reduce activity, let heat dissipate. In biological terms, this is rest. In computational terms, this is throttling. The physics is the same: when you are overheating, the only fix is to reduce heat generation and wait for equilibration.

The "sleep" mode is interesting—reflective boundary, zero action. This is the computational equivalent of lying still in a cool room. Maximum passive cooling, minimum heat generation.
:::



(node-70-metabolic)=
**Node 70: MetricFadingCheck**

*Trigger condition:* $f(B/B_{\text{crit}}) < \epsilon_{\text{fade}}$ (metric severely degraded)

*Interpretation:* The latent geometry has collapsed below functional resolution. The agent is effectively hallucinating.

*Remediation:*
1. Prioritize energy harvesting (Node 68 interventions)
2. Reduce control authority (don't trust faded-metric decisions)
3. Fall back to reflexive/hardcoded behaviors
4. Signal distress to external systems (if available)

:::{div} feynman-prose
Node 70 catches a subtler failure mode. The agent might still have some battery left, but if the metric has faded too far, it cannot make good decisions anymore. Its representations are too blurry to distinguish good actions from bad ones.

The key remediation is: **do not trust your own judgments**. When metric fading is severe, the agent should fall back to reflexive behaviors—hardcoded heuristics that do not require fine discrimination. It is the computational equivalent of "when in doubt, hold still and wait for help."

Signaling distress to external systems acknowledges that an agent in severe metric fading may not be able to save itself. If there are external resources available—a supervisor, a safety system, a caretaker—this is the time to invoke them.
:::



(sec-summary-closed-thermodynamic-loop)=
## Summary: The Closed Thermodynamic Loop

With the Metabolic Transducer, the Fragile Agent becomes a thermodynamically closed system—an **autopoietic machine** whose existence depends on its own successful operation.

:::{div} feynman-prose
Let us step back and see what we have built.

We started with a puzzle: where does the energy for computation come from? The Landauer bound tells us that thinking costs energy, but it does not tell us where to get it.

The answer, it turns out, comes from Szilard's analysis of Maxwell's Demon. Information about low-entropy configurations can be converted to work at the rate $k_B T$ per nat. The reward signal encodes exactly this kind of information—it tells the agent where the resources are. The Metabolic Transducer converts this information to usable energy.

The agent now forms a closed thermodynamic loop:
1. **Harvest**: Find resources (positive reward), extract work via Szilard engine
2. **Store**: Accumulate energy in the internal battery
3. **Spend**: Use energy for inference (Landauer cost) and existence (basal rate)
4. **Repeat**: Or die if the balance goes negative

The Autopoietic Inequality is the survival condition: harvest more than you spend. The Fading Metric Law is what happens when you fail: your geometry collapses, your distinctions blur, and you drift into hallucination.

This is not just a theoretical framework—it is a design specification for agents that must operate under thermodynamic constraints. Standard RL is the limiting case where energy is free and unlimited. Real agents, biological or artificial, must play by stricter rules.
:::

**Summary Table: Autopoietic Thermodynamics**

| Quantity | Symbol | Dimension | Definition | Physics Analogue |
|:---------|:-------|:----------|:-----------|:-----------------|
| Internal Battery | $B(t)$ | $[\text{Energy}]$ | {prf:ref}`def-internal-battery` | ATP reservoir |
| Metabolic Transducer | $\mathfrak{T}$ | $[\text{Power}]$ | {prf:ref}`def-metabolic-transducer` | Mitochondria |
| Metabolic Cost | $\dot{\mathcal{M}}$ | $[\text{Power}]$ | {prf:ref}`thm-generalized-landauer-bound` | Basal metabolism |
| Metric Scaling | $f(B)$ | $[1]$ | {prf:ref}`thm-fading-metric-law` | Neural SNR |
| Carnot Efficiency | $\eta$ | $[1]$ | {prf:ref}`thm-carnot-transduction-bound` | Heat engine efficiency |
| Homeostatic Potential | $\Phi_{\text{homeo}}$ | $[\text{nats}]$ | {prf:ref}`def-homeostatic-potential` | Hunger drive |
| Critical Energy | $B_{\text{crit}}$ | $[\text{Energy}]$ | {prf:ref}`thm-fading-metric-law` | Metabolic threshold |

**The Complete Energy Flow:**

```

   Environment --[Observation]---> Inference --[Action]---> Environment
        |                            |                         |
        |                            |                         |
        v                            v                         |
    +-------+                  +----------+                    |
    |Reward |                  |Metabolic |                    |
    |Signal |                  |Cost M(t) |                    |
    +---+---+                  +----+-----+                    |
        |                           |                          |
        v                           v                          |
   +----------+              +-----------+                     |
   |Transducer|              |  Landauer |                     |
   |  T(r)    |              |   Bound   |                     |
   +----+-----+              +-----+-----+                     |
        |                          |                           |
        |     +---------+          |                           |
        +---->| Battery |<---------+                           |
              |  B(t)   |                                      |
              +----+----+                                      |
                   |                                           |
                   v                                           |
              +---------+                                      |
              | Fading  |                                      |
              | Metric  |<-------------------------------------+
              | G_eff   |
              +---------+

```

**Cross-references:**
- Closes the loop opened in {ref}`sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics` (Metabolism)
- Connects the Reward Field ({ref}`sec-the-reward-field-value-forms-and-hodge-geometry`) to agent survival
- Provides the ultimate boundary condition for the Universal Governor ({ref}`sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller`)
- Adds metabolic viability constraint to the Parameter Space Sieve ({ref}`sec-parameter-space-sieve`)

**The Autopoietic Closure:** The agent's objective function is derived from the structural necessity of maintaining $B(t) > 0$ while minimizing free energy of the task. Survival is not externally specified but emerges from the thermodynamic constraint: insufficient harvesting leads to metric collapse (Theorem {prf:ref}`thm-fading-metric-law`) and loss of computational capacity.

:::{div} feynman-prose
And there you have it. The agent is not just a learning machine—it is a thermodynamic entity, a dissipative structure that maintains itself against the relentless tide of entropy. Its goals are not arbitrary preferences but emerge from the physical necessity of survival.

This closes the loop we opened in the metabolism chapter. We now have a complete picture: thinking costs energy (Landauer), correct prediction provides energy (Szilard), and the balance between them determines survival (Autopoiesis). The geometry itself depends on this balance (Fading Metric), so that a dying agent cannot even think clearly about how to save itself.

It is a harsh picture in some ways. But it is also, I think, a beautiful one. The agent is not separate from the physics—it is made of physics, constrained by physics, and can only survive by understanding physics. Even the question "what should I value?" has a thermodynamic answer: value what keeps you alive long enough to keep valuing things.

That is not the whole story of value, of course. But it is the foundation—the bedrock on which everything else must be built.
:::
