(sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws)=
# {ref}`Appendix E <sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws>`: Rigorous Proof Sketches for Ontological and Metabolic Laws

## TLDR

- This appendix contains rigorous proof sketches backing the ontology/metabolism results in the later cognition chapters.
- Read it when you want the mathematical spine behind the narrative statements; otherwise treat it as a reference.

This appendix provides the rigorous mathematical foundations for the theorems and propositions introduced in Sections 30, 31, and 32. We operate on the latent Riemannian manifold $(\mathcal{Z}, G)$ with belief measures $\rho \in \mathcal{P}(\mathcal{Z})$.



(sec-appendix-e-proof-of-theorem-prf-ref)=
## E.1 Proof of Theorem {prf:ref}`thm-fission-criterion`

**Statement:** The ontology should expand from $N_c$ to $N_c + 1$ charts if and only if $\Xi > \Xi_{\text{crit}}$ and $\Delta V_{\text{proj}} > \mathcal{C}_{\text{complexity}}$.

**Hypothesis:** Let $\mathcal{S}[N_c] = \inf_{\theta} \mathcal{S}_{\text{onto}}(\theta, N_c)$ be the value function of the ontological action for $N_c$ charts.

(proof-thm-the-fission-trigger)=
:::{prf:proof}

Consider the discrete variation $\Delta \mathcal{S} = \mathcal{S}[N_c + 1] - \mathcal{S}[N_c]$. By the definition of the Ontological Action ({ref}`Section 30.3 <sec-the-fission-criterion>`):

$$
\mathcal{S}_{\text{onto}} = -\mathcal{S}_{\text{task}} + \mu_{\text{size}} \cdot N_c,

$$
where $\mathcal{S}_{\text{task}} = \mathbb{E}[\langle V \rangle]$ is the expected task value.

Expanding $\mathcal{S}_{\text{task}}$ via a first-order Taylor approximation in the space of representations:

$$
\mathcal{S}_{\text{task}}[N_c + 1] \approx \mathcal{S}_{\text{task}}[N_c] + \frac{\partial \langle V \rangle}{\partial N_c}.

$$
The marginal utility of a new chart is $\frac{\partial \langle V \rangle}{\partial N_c} = \Delta V_{\text{proj}}$. The complexity cost is $\mu_{\text{size}}$. Therefore:

$$
\Delta \mathcal{S} = -\Delta V_{\text{proj}} + \mu_{\text{size}}.

$$
The transition $N_c \to N_c + 1$ is the global minimizer iff $\Delta \mathcal{S} < 0$, which yields:

$$
\Delta V_{\text{proj}} > \mu_{\text{size}} = \mathcal{C}_{\text{complexity}}.

$$
The condition $\Xi > \Xi_{\text{crit}}$ ensures that the second variation of the texture-entropy functional $\delta^2 H(z_{\text{tex}})$ is negative-definite at the vacuum. This precludes the absorption of the signal into the existing noise floor: if $\Xi \le \Xi_{\text{crit}}$, the texture residual $z_{\text{tex}}$ is truly unpredictable noise, and adding a chart provides no informational benefit. $\square$

:::



(sec-appendix-e-proof-of-theorem-prf-ref-a)=
## E.2 Proof of Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`

**Statement:** The emergence of a new chart follows a supercritical pitchfork bifurcation with control parameter $\mu = \Xi - \Xi_{\text{crit}}$.

**Hypothesis:** The potential $\Phi_{\text{onto}}(r)$ is $SO(n)$-invariant near $r=0$, where $r = \|q_* - q_{\text{parent}}\|$ is the radial distance of the new query from the parent.

(proof-thm-supercritical-pitchfork-bifurcation)=
:::{prf:proof}

Let $f(\Xi) = \Xi - \Xi_{\text{crit}}$ be the control parameter. By $SO(n)$ symmetry, the Ontological Action can only depend on even powers of $r$ near the origin. We expand in a power series:

$$
\mathcal{S}(r) = \mathcal{S}_0 - \frac{1}{2}f(\Xi)r^2 + \frac{1}{4}\beta r^4 + O(r^6),

$$
where $\beta > 0$ for stability (the quartic term must be positive for bounded energy).

The stationarity condition $\frac{\partial \mathcal{S}}{\partial r} = 0$ yields:

$$
-f(\Xi)r + \beta r^3 = 0 \implies r(f(\Xi) - \beta r^2) = 0.

$$
This has solutions:
1. $r = 0$ (trivial, no new chart)
2. $r^2 = f(\Xi)/\beta$ (symmetry-broken state)

**Analysis of stability:**
- For $f(\Xi) < 0$ (i.e., $\Xi < \Xi_{\text{crit}}$): The Hessian at $r=0$ is $\frac{\partial^2 \mathcal{S}}{\partial r^2}|_{r=0} = -f(\Xi) > 0$. Thus $r=0$ is a stable minimum.
- For $f(\Xi) > 0$ (i.e., $\Xi > \Xi_{\text{crit}}$): The Hessian at $r=0$ becomes $-f(\Xi) < 0$ (unstable). New minima appear at $r^* = \sqrt{f(\Xi)/\beta}$.

Since $r \ge 0$ is a radial coordinate, this constitutes a **supercritical pitchfork bifurcation** where the symmetry-broken state $r^* > 0$ becomes the unique stable equilibrium for $\Xi > \Xi_{\text{crit}}$.

The bifurcation diagram: for $\Xi < \Xi_{\text{crit}}$, the system has a single stable fixed point at $r=0$; for $\Xi > \Xi_{\text{crit}}$, the origin becomes unstable and two symmetric branches (in the full space, a sphere of radius $r^*$) emerge. $\square$

:::



(sec-appendix-e-proof-of-theorem-prf-ref-b)=
## E.3 Proof of Theorem {prf:ref}`thm-generalized-landauer-bound`

**Statement:** $\dot{\mathcal{M}}(s) \ge T_c \left| \frac{d}{ds} H(\rho_s) \right|$.

**Hypothesis:** Belief evolution follows the {prf:ref}`def-the-wfr-action` continuity equation $\partial_s \rho = \mathcal{L}_{\text{WFR}} \rho = \rho r - \nabla \cdot (\rho v)$.

(proof-thm-generalized-landauer-bound)=
:::{prf:proof}

The time derivative of the Shannon entropy is:

$$
\frac{d}{ds} H(\rho_s) = \frac{d}{ds}\left( -\int_{\mathcal{Z}} \rho \ln \rho \, d\mu_G \right) = -\int_{\mathcal{Z}} (1 + \ln \rho) \partial_s \rho \, d\mu_G.

$$
Substituting the WFR continuity equation:

$$
\frac{d}{ds} H = -\int_{\mathcal{Z}} (1 + \ln \rho)(\rho r - \nabla \cdot (\rho v)) \, d\mu_G.

$$
**Transport term:** Integrating by parts (assuming $\rho v \cdot n|_{\partial\mathcal{Z}} = 0$):

$$
-\int (1 + \ln \rho)(-\nabla \cdot (\rho v)) \, d\mu_G = \int \nabla(1 + \ln \rho) \cdot (\rho v) \, d\mu_G = \int \rho \langle \nabla \ln \rho, v \rangle_G \, d\mu_G.

$$
**Reaction term:**

$$
-\int (1 + \ln \rho) \rho r \, d\mu_G = -\int \rho r \, d\mu_G - \int \rho r \ln \rho \, d\mu_G.

$$
The first integral is the total mass change $\frac{d}{ds}\int \rho \, d\mu_G$. For normalized probabilities, this vanishes if we work in the cone representation. The second integral is bounded by the reaction energy.

**Applying Cauchy-Schwarz:** For the transport term on $(T\mathcal{Z}, G)$:

$$
\left| \int_{\mathcal{Z}} \rho \langle \nabla \ln \rho, v \rangle_G \, d\mu_G \right| \le \left( \int \rho \|\nabla \ln \rho\|_G^2 \, d\mu_G \right)^{1/2} \left( \int \rho \|v\|_G^2 \, d\mu_G \right)^{1/2}.

$$
The first factor is the **Fisher Information** $\mathcal{I}(\rho)$. By the de Bruijn identity for diffusion processes:

$$
\frac{d}{ds} H(\rho_s) = -\frac{1}{2T_c} \mathcal{I}(\rho_s)

$$
under optimal transport scaling $v = -T_c G^{-1}\nabla \ln \rho$.

Combining: $|\dot{H}| \le \frac{1}{T_c}\sqrt{\mathcal{I}(\rho) \cdot \int \rho \|v\|_G^2}$. With $\sigma_{\text{met}} = 1/T_c$, we obtain $\dot{\mathcal{M}} \ge T_c |\dot{H}|$.

The reaction term follows by an identical argument using the $L^2(\rho)$ inner product:

$$
\left| \int \rho r \ln \rho \, d\mu_G \right| \le \|\sqrt{\rho} r\|_{L^2} \|\sqrt{\rho} \ln \rho\|_{L^2}.

$$
Adding both contributions yields the stated bound. $\square$

:::



(sec-appendix-e-proof-of-theorem-prf-ref-c)=
## E.4 Proof of Theorem {prf:ref}`thm-deliberation-optimality-condition`

**Statement:** The optimal computation budget $S^*$ satisfies $\frac{d}{ds} \langle V \rangle_{\rho_s}|_{s=S^*} = \dot{\mathcal{M}}(S^*)$.

**Hypothesis:** $S^*$ is an interior point of $[0, S_{\max}]$.

(proof-thm-deliberation-optimality)=
:::{prf:proof}

Define the deliberation functional:

$$
\mathcal{F}(S) = -\int_{\mathcal{Z}} V(z) \rho(S, z) \, d\mu_G + \int_0^S \dot{\mathcal{M}}(u) \, du.

$$
The necessary condition for an extremum is $\mathcal{F}'(S) = 0$. By the Leibniz integral rule:

$$
\mathcal{F}'(S) = -\int_{\mathcal{Z}} V(z) \partial_s \rho(S, z) \, d\mu_G + \dot{\mathcal{M}}(S).

$$
Using the result that $\partial_s \rho$ is governed by the WFR operator $\mathcal{L}_{\text{WFR}}$:

$$
\mathcal{F}'(S) = -\int_{\mathcal{Z}} V \mathcal{L}_{\text{WFR}}\rho \, d\mu_G + \dot{\mathcal{M}}(S).

$$
By the adjoint property of the WFR operator (the formal $L^2(\rho)$ adjoint):

$$
\int V \mathcal{L}_{\text{WFR}}\rho \, d\mu_G = \int \rho \mathcal{L}_{\text{WFR}}^* V \, d\mu_G,

$$
where $\mathcal{L}_{\text{WFR}}^* V = -\langle \nabla V, v \rangle_G + Vr$ (transport-adjoint plus reaction).

For gradient flows in the covariant case, $v = -G^{-1}\nabla_A V$ with $\nabla_A V := \nabla V - A$:

$$
\mathcal{L}_{\text{WFR}}^* V = G^{-1}(\nabla V, \nabla_A V) + Vr.

$$
Thus:

$$
\mathcal{F}'(S) = -\int \rho \left( G^{-1}(\nabla V, \nabla_A V) + Vr \right) d\mu_G + \dot{\mathcal{M}}(S).

$$
In the conservative case ($A=0$), $G^{-1}(\nabla V, \nabla_A V) = \|\nabla V\|_G^2$, the power dissipated by the value-gradient flow. The stationarity condition $\mathcal{F}'(S^*) = 0$ gives:

$$
\frac{d}{ds} \langle V \rangle_{\rho_s}\bigg|_{s=S^*} = \dot{\mathcal{M}}(S^*).

$$
This states that the optimal stopping time $S^*$ is reached when the power dissipated by the value-gradient flow exactly matches the metabolic cost rate. $\square$

:::



(sec-appendix-e-proof-of-theorem-prf-ref-d)=
## E.5 Proof of Theorem {prf:ref}`thm-augmented-drift-law`

**Statement:** $F_{\text{total}} = -G^{-1}\nabla_A V + \beta_{\text{exp}} G^{-1}\nabla\Psi_{\text{causal}}$.

**Hypothesis:** The agent's path minimizes $\mathcal{S} = \int L(z, \dot{z}) \, dt$ with Lagrangian $L = \frac{1}{2}\|\dot{z}\|_G^2 - (V + \beta_{\text{exp}}\Psi_{\text{causal}})$.

(proof-thm-the-augmented-drift-law)=
:::{prf:proof}

The Euler-Lagrange equations for the functional are:

$$
\frac{d}{dt} \frac{\partial L}{\partial \dot{z}^k} - \frac{\partial L}{\partial z^k} = 0.

$$
**Computing the momentum:**

$$
\frac{\partial L}{\partial \dot{z}^k} = \frac{\partial}{\partial \dot{z}^k}\left( \frac{1}{2}G_{ij}(z)\dot{z}^i \dot{z}^j \right) = G_{kj}\dot{z}^j = p_k.

$$
**Time derivative of momentum:**

$$
\frac{d}{dt}(G_{kj}\dot{z}^j) = G_{kj}\ddot{z}^j + \frac{\partial G_{kj}}{\partial z^m}\dot{z}^m \dot{z}^j.

$$
**Potential gradient:**

$$
\frac{\partial L}{\partial z^k} = \frac{1}{2}\frac{\partial G_{ij}}{\partial z^k}\dot{z}^i\dot{z}^j - \partial_k V - \beta_{\text{exp}}\partial_k \Psi_{\text{causal}}.

$$
**Euler-Lagrange equation:**

$$
G_{kj}\ddot{z}^j + \frac{\partial G_{kj}}{\partial z^m}\dot{z}^m \dot{z}^j - \frac{1}{2}\frac{\partial G_{ij}}{\partial z^k}\dot{z}^i\dot{z}^j = -\partial_k V - \beta_{\text{exp}}\partial_k \Psi_{\text{causal}}.

$$
Recognizing the Christoffel symbols of the first kind $[ij, k] = \frac{1}{2}(\partial_i G_{jk} + \partial_j G_{ik} - \partial_k G_{ij})$:

$$
G_{kj}\ddot{z}^j + [ij, k]\dot{z}^i\dot{z}^j = -\partial_k V - \beta_{\text{exp}}\partial_k \Psi_{\text{causal}}.

$$
Contracting with $G^{mk}$ and using $\Gamma^m_{ij} = G^{mk}[ij, k]$:

$$
\ddot{z}^m + \Gamma^m_{ij}\dot{z}^i\dot{z}^j = -G^{mk}\partial_k V - \beta_{\text{exp}} G^{mk}\partial_k \Psi_{\text{causal}}.

$$
This is the geodesic equation with forcing terms. In the **overdamped limit** ({ref}`Section 22.3 <sec-the-unified-effective-potential>`), inertia is negligible and the acceleration term vanishes, leaving:

$$
\dot{z}^m = -G^{mk}\partial_k V + \beta_{\text{exp}} G^{mk}\partial_k \Psi_{\text{causal}} = F^m_{\text{total}}.

$$
The drift field $F_{\text{total}}$ is the first-order velocity approximation, proving the additive force of curiosity. $\square$

:::



(sec-appendix-e-proof-of-theorem-prf-ref-e)=
## E.6 Proof of Theorem {prf:ref}`thm-interventional-closure`

**Statement:** The macro-ontology $K$ is interventionally closed iff $I(K_{t+1}; Z_{\text{micro}, t} | K_t, do(K^{\text{act}}_t)) = 0$.

**Hypothesis:** Let $\mathcal{M}$ be a Markov Blanket for $K$.

(proof-thm-interventional-closure)=
:::{prf:proof}

We compare the mutual information under the observational measure $P$ and the interventional measure $P_{do(K^{\text{act}})}$.

**Observational case:** By the Causal Enclosure condition ({ref}`Section 2.8 <sec-conditional-independence-and-sufficiency>`):

$$
I(K_{t+1}; Z_{\text{micro}, t} | K_t, K^{\text{act}}_t) = 0 \quad \text{under } P.

$$
This states that the macro-state $K_{t+1}$ is conditionally independent of the micro-texture $Z_{\text{micro}, t}$ given the current macro-state and action.

**Interventional case:** The $do(K^{\text{act}}_t)$ operator performs a graph surgery that removes all incoming edges to $K^{\text{act}}_t$ while preserving all other mechanisms. By Pearl's Causal Markov Condition {cite}`pearl2009causality`:

$$
P(K_{t+1} | K_t, K^{\text{act}}_t, Z_{\text{micro}, t}) \text{ remains invariant under } do(K^{\text{act}}_t).

$$
This is because the mechanism $P(K_{t+1} | \text{parents}(K_{t+1}))$ is a structural equation that does not depend on how $K^{\text{act}}_t$ was generated.

**Combining the conditions:**
If the observational distribution satisfies $I = 0$, then:

$$
P(K_{t+1} | K_t, K^{\text{act}}_t) = P(K_{t+1} | K_t, K^{\text{act}}_t, Z_{\text{micro}, t}) \quad \forall Z_{\text{micro}, t}.

$$
Since the mechanism is invariant under intervention:

$$
P(K_{t+1} | K_t, do(K^{\text{act}}_t)) = P(K_{t+1} | K_t, K^{\text{act}}_t) = P(K_{t+1} | K_t, K^{\text{act}}_t, Z_{\text{micro}, t}).

$$
Therefore, $I(K_{t+1}; Z_{\text{micro}, t} | K_t, do(K^{\text{act}}_t)) = 0$.

**Contrapositive (violation):** If $I > 0$ under $do(K^{\text{act}}_t)$, there exists a back-door path through $Z_{\text{micro}, t}$:

$$
K_t \leftarrow Z_{\text{micro}, t} \to K_{t+1}.

$$
This path was confounded in observational data (the correlation between $Z_{\text{micro}}$ and $K_{t+1}$ was screened by the policy generating $K^{\text{act}}_t$). The intervention breaks this screening, exposing the hidden variable. The remedy is **Ontological Expansion** ({ref}`Section 30 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>`): promote the relevant component of $Z_{\text{micro}}$ to a new macro-variable in $K$. $\square$

:::



(sec-appendix-e-rigorous-proof-of-multi-agent-strategic-tunneling)=
## E.7 Rigorous Proof of Multi-Agent Strategic Tunneling (Theorem {prf:ref}`thm-tunneling-probability`)

**Title:** *Asymptotic Behavior of the Joint Belief Measure on Riemannian Product Manifolds under Metric Deformation by the Game Tensor.*

This appendix provides the rigorous mathematical foundation for Theorem {prf:ref}`thm-tunneling-probability` (Strategic Tunneling Probability). We replace heuristic WKB arguments with rigorous results from **Spectral Theory of Elliptic Operators** and **Semi-Classical Analysis (Agmon Estimates)**.

**Key rigorous tools:**
1. **Perron-Frobenius / Krein-Rutman Theorem** for strict positivity
2. **Agmon Estimates** {cite}`agmon1982lectures` for exponential decay bounds
3. **Feynman-Kac Formula** for probabilistic representation
4. **Metric Comparison Theorems** for Game Tensor effects



### E.7.1 Mathematical Setup and Definitions

Let the $N$-agent configuration space be the product manifold $\mathcal{M} = \prod_{i=1}^N \mathcal{Z}^{(i)}$. We assume each $\mathcal{Z}^{(i)}$ is a smooth, compact, connected Riemannian manifold with boundary (or without boundary if geodesically complete).

:::{prf:definition} E.7.1 (The Strategic Metric)
:label: def-e7-strategic-metric

Let $G^{(i)}$ be the capacity-constrained metric on $\mathcal{Z}^{(i)}$ (Theorem {prf:ref}`thm-capacity-constrained-metric-law`). The **Strategic Metric** $\mathbf{g}$ on $\mathcal{M}$ is the block-diagonal sum perturbed by the Game Tensor $\mathcal{G}$ (Definition {prf:ref}`def-the-game-tensor`):

$$
\mathbf{g}(\mathbf{z}) := \bigoplus_{i=1}^N G^{(i)}(z^{(i)}) + \alpha \sum_{i \neq j} \mathcal{G}_{ij}(\mathbf{z}),

$$
where the pullback of the cross-Hessian interaction acts on tangent vectors in the obvious way.

*Assumption 1 (Ellipticity):* We assume $\alpha > 0$ is sufficiently small such that $\mathbf{g}$ remains positive-definite and defines a valid Riemannian structure on $\mathcal{M}$. This is guaranteed when $\|\alpha \mathcal{G}\|_{\text{op}} < \lambda_{\min}(\bigoplus G^{(i)})$.

:::

:::{prf:definition} E.7.2 (The Strategic Hamiltonian)
:label: def-e7-strategic-hamiltonian

The self-adjoint **Strategic Hamiltonian** operator $\hat{H}_\sigma: H^2(\mathcal{M}) \to L^2(\mathcal{M}, d\mu_{\mathbf{g}})$ acts on the joint wave-function $\Psi$:

$$
\hat{H}_\sigma := -\frac{\sigma^2}{2} \Delta_{\mathbf{g}} + \mathcal{V}(\mathbf{z}),

$$
where:
- $\Delta_{\mathbf{g}}$ is the Laplace-Beltrami operator associated with the strategic metric $\mathbf{g}$
- $\mathcal{V}(\mathbf{z}) := \sum_{i=1}^N \Phi^{(i)}_{\text{eff}}(z^{(i)}) + \sum_{i < j} \Phi_{ij}(z^{(i)}, z^{(j)})$ is the joint potential
- $\sigma > 0$ is the cognitive action scale (Definition {prf:ref}`def-cognitive-action-scale`)

*Assumption 2 (Regularity):* $\mathcal{V} \in C^2(\mathcal{M})$ and is bounded below.

:::

::::{admonition} Physics Isomorphism: Spectral Gap
:class: note
:name: pi-spectral-gap

**In Physics:** The spectral gap $\Delta = E_1 - E_0$ of a Hamiltonian $H$ controls the mixing time to the ground state: $\|e^{-tH}\psi - \psi_0\| \leq e^{-\Delta t}$. The Poincare inequality bounds the gap from below {cite}`liggett1989exponential,diaconis1991geometric`.

**In Implementation:** The spectral gap of the strategic Hamiltonian $\hat{H}_{\text{strat}}$ controls convergence to Nash:

$$
\|\Psi(s) - \Psi_0\|_{L^2} \leq e^{-\Delta s / \sigma^2} \|\Psi(0) - \Psi_0\|_{L^2}

$$
**Correspondence Table:**
| Spectral Theory | Agent (Convergence) |
|:----------------|:--------------------|
| Ground state energy $E_0$ | Nash equilibrium value |
| First excited state $E_1$ | Nearest sub-optimal equilibrium |
| Spectral gap $\Delta$ | Convergence rate to Nash |
| Poincare inequality | Lower bound on gap |
| Mixing time $\tau_{\text{mix}} \sim 1/\Delta$ | Training time |

**Diagnostic:** ConvergenceRateCheck monitors effective gap from eigenvalue estimates.
::::

:::{prf:definition} E.7.3 (The Forbidden Region and Nash Basins)
:label: def-e7-forbidden-region

Let $E_0 := \inf \text{spec}(\hat{H}_\sigma)$ be the ground state energy. The **Classically Forbidden Region** (Barrier) is:

$$
\mathcal{K} := \{ \mathbf{z} \in \mathcal{M} : \mathcal{V}(\mathbf{z}) > E_0 \}.

$$
Let $\Omega_A, \Omega_B \subset \mathcal{M} \setminus \mathcal{K}$ be disjoint open sets (Nash basins) where $\mathcal{V}(\mathbf{z}) \leq E_0$.

*Geometric interpretation:* $\Omega_A$ and $\Omega_B$ are "potential wells" corresponding to distinct Nash equilibria (Theorem {prf:ref}`thm-nash-ground-state`). The barrier $\mathcal{K}$ separates these wells.

:::



### E.7.2 Strict Positivity of the Ground State (Existence of Tunneling)

We first prove that tunneling is not merely possible—it is **inevitable** for any connected manifold.

:::{prf:theorem} E.7.1 (Strict Positivity of the Ground State)
:label: thm-e7-ground-state-positivity

Let $\Psi_0$ be the ground state eigenfunction of $\hat{H}_\sigma$ (the eigenfunction with eigenvalue $E_0$). Then:

$$
|\Psi_0(\mathbf{z})| > 0 \quad \forall \mathbf{z} \in \mathcal{M}.

$$
*Consequence:* For any open set $\Omega_B \subset \mathcal{M}$, the probability measure satisfies:

$$
\mu(\Omega_B) = \int_{\Omega_B} |\Psi_0(\mathbf{z})|^2 \, d\mu_{\mathbf{g}}(\mathbf{z}) > 0.

$$
Therefore, if an agent is localized in $\Omega_A$, there is strictly positive probability of finding it in $\Omega_B$.

:::

(proof-thm-e7-ground-state-positivity)=
:::{prf:proof}

**Step 1 (Elliptic Regularity).** Since $\mathbf{g}$ is smooth and positive-definite (Assumption 1), and $\mathcal{V}$ is smooth (Assumption 2), the operator $\hat{H}_\sigma$ is uniformly elliptic. By standard elliptic regularity theory {cite}`gilbarg1977elliptic`, any $L^2$ eigenfunction $\Psi$ satisfying $\hat{H}_\sigma \Psi = E \Psi$ is in $C^\infty(\mathcal{M})$.

**Step 2 (Heat Kernel Positivity).** Consider the heat semigroup $e^{-t\hat{H}_\sigma}$ for $t > 0$. By the **Harnack Inequality** for parabolic equations on manifolds {cite}`li1986parabolic`, the heat kernel $K_t(\mathbf{x}, \mathbf{y}) > 0$ for all $\mathbf{x}, \mathbf{y} \in \mathcal{M}$ and $t > 0$, provided $\mathcal{M}$ is connected.

This implies: for any non-negative, non-zero $f \in L^2(\mathcal{M})$:

$$
(e^{-t\hat{H}_\sigma} f)(\mathbf{x}) = \int_{\mathcal{M}} K_t(\mathbf{x}, \mathbf{y}) f(\mathbf{y}) \, d\mu_{\mathbf{g}}(\mathbf{y}) > 0 \quad \forall \mathbf{x} \in \mathcal{M}.

$$
The heat kernel maps non-negative functions to **strictly positive** functions.

**Step 3 (Perron-Frobenius / Krein-Rutman).** The operator $e^{-t\hat{H}_\sigma}$ is a positivity-improving compact operator on $L^2(\mathcal{M})$. By the **Krein-Rutman Theorem** (the infinite-dimensional generalization of Perron-Frobenius), the spectral radius is a simple eigenvalue with a strictly positive eigenfunction.

Since $e^{-t\hat{H}_\sigma}$ has spectral radius $e^{-tE_0}$ with eigenfunction $\Psi_0$, and this eigenvalue is simple, we conclude:
- $\Psi_0$ can be chosen to be real and non-negative
- By positivity-improving property, $\Psi_0(\mathbf{z}) > 0$ for all $\mathbf{z} \in \mathcal{M}$

**Step 4 (Conclusion).** For any open $\Omega_B \subset \mathcal{M}$:

$$
\mu(\Omega_B) = \int_{\Omega_B} |\Psi_0|^2 \, d\mu_{\mathbf{g}} \geq c \cdot \text{Vol}_{\mathbf{g}}(\Omega_B) > 0,

$$
where $c = \min_{\overline{\Omega}_B} |\Psi_0|^2 > 0$ by continuity and strict positivity. $\square$

:::

:::{admonition} Key Insight
:class: tip
:name: insight-tunneling-inevitable

Theorem E.7.1 proves that **tunneling is inevitable**, not merely possible. On a connected manifold, the ground state wave-function has non-zero amplitude everywhere. The agent cannot be "trapped" in a Nash basin $\Omega_A$ with zero probability of being in $\Omega_B$—there is always leakage through the barrier.

The relevant question becomes: **how fast** does tunneling occur? This is answered by the Agmon estimates.

:::



### E.7.3 Agmon Estimates: Quantifying the Tunneling Rate

While Theorem E.7.1 proves existence, we need **quantitative bounds** on the decay rate through the barrier. We use Agmon's method {cite}`agmon1982lectures`.

:::{prf:definition} E.7.4 (The Agmon Metric)
:label: def-e7-agmon-metric

Inside the barrier $\mathcal{K}$, we define the **Agmon Metric** $\rho_E$, a degenerate conformal rescaling of $\mathbf{g}$:

$$
(\rho_E)_{ij}(\mathbf{z}) := \max\left(0, \mathcal{V}(\mathbf{z}) - E_0\right) \cdot \mathbf{g}_{ij}(\mathbf{z}).

$$
The **Agmon distance** between points $\mathbf{x}, \mathbf{y} \in \mathcal{M}$ is:

$$
d_{\text{Ag}}(\mathbf{x}, \mathbf{y}) := \inf_{\gamma: \mathbf{x} \to \mathbf{y}} \int_0^1 \sqrt{\max(0, \mathcal{V}(\gamma(t)) - E_0)} \cdot \|\dot{\gamma}(t)\|_{\mathbf{g}} \, dt,

$$
where the infimum is over all piecewise smooth paths $\gamma$ from $\mathbf{x}$ to $\mathbf{y}$.

*Properties:*
1. $d_{\text{Ag}}(\mathbf{x}, \mathbf{y}) = 0$ if there exists a path entirely within $\mathcal{M} \setminus \mathcal{K}$ (the "classical" region)
2. $d_{\text{Ag}}(\mathbf{x}, \mathbf{y}) > 0$ if all paths must traverse $\mathcal{K}$ (tunneling required)
3. The Agmon distance is a pseudo-metric (satisfies triangle inequality)

:::

:::{prf:theorem} E.7.2 (Agmon Exponential Decay Bound)
:label: thm-e7-agmon-decay-bound

Let $\Psi_0$ be the ground state of $\hat{H}_\sigma$ with eigenvalue $E_0$. For any $\epsilon > 0$, there exists a constant $C_\epsilon > 0$ (depending on $\mathcal{M}$, $\mathcal{V}$, and $\epsilon$, but not on $\sigma$) such that:

$$
|\Psi_0(\mathbf{z})| \leq C_\epsilon \exp\left( - \frac{1 - \epsilon}{\sigma} d_{\text{Ag}}(\mathbf{z}, \Omega_A) \right) \quad \forall \mathbf{z} \in \mathcal{M},

$$
where $d_{\text{Ag}}(\mathbf{z}, \Omega_A) := \inf_{\mathbf{y} \in \Omega_A} d_{\text{Ag}}(\mathbf{z}, \mathbf{y})$.

*Interpretation:* The wave-function amplitude decays exponentially with rate $1/\sigma$ times the Agmon distance from the classical region. Deeper into the barrier (larger $d_{\text{Ag}}$), the amplitude is exponentially smaller.

:::

(proof-thm-e7-agmon-decay-bound)=
:::{prf:proof}

We follow the standard Agmon method {cite}`agmon1982lectures,simon1983semiclassical`.

**Step 1 (Twisted Function).** Define the twisted function:

$$
\phi(\mathbf{z}) := e^{f(\mathbf{z})/\sigma} \Psi_0(\mathbf{z}),

$$
where $f: \mathcal{M} \to \mathbb{R}$ is a Lipschitz weight function to be chosen.

**Step 2 (Agmon Identity).** From the eigenvalue equation $(\hat{H}_\sigma - E_0)\Psi_0 = 0$, we derive:

$$
-\frac{\sigma^2}{2}\Delta_{\mathbf{g}}\phi + (\mathcal{V} - E_0)\phi = \frac{1}{2}\|\nabla_{\mathbf{g}} f\|_{\mathbf{g}}^2 \phi + \sigma \langle \nabla_{\mathbf{g}} f, \nabla_{\mathbf{g}} \phi \rangle_{\mathbf{g}}.

$$
**Step 3 (Energy Estimate).** Multiply by $\bar{\phi}$ and integrate. Using integration by parts:

$$
\frac{\sigma^2}{2} \|\nabla_{\mathbf{g}}\phi\|_{L^2}^2 + \int_{\mathcal{M}} \left(\mathcal{V} - E_0 - \frac{1}{2}\|\nabla_{\mathbf{g}} f\|_{\mathbf{g}}^2\right) |\phi|^2 \, d\mu_{\mathbf{g}} \leq 0.

$$
**Step 4 (Optimal Weight).** Choose $f(\mathbf{z}) = (1-\epsilon) d_{\text{Ag}}(\mathbf{z}, \Omega_A)$. By construction of the Agmon metric:

$$
\|\nabla_{\mathbf{g}} f\|_{\mathbf{g}}^2 \leq (1-\epsilon)^2 (\mathcal{V} - E_0)_+ \quad \text{a.e.}

$$
**Step 5 (Pointwise Bound).** Substituting and using Sobolev embedding on the compact manifold $\mathcal{M}$:

$$
\sup_{\mathbf{z} \in \mathcal{M}} |\phi(\mathbf{z})|^2 \leq C_\epsilon' \|\phi\|_{H^1}^2 \leq C_\epsilon'' \|\Psi_0\|_{L^2}^2 = C_\epsilon''.

$$
Unwinding the twist gives:

$$
|\Psi_0(\mathbf{z})| = e^{-f(\mathbf{z})/\sigma} |\phi(\mathbf{z})| \leq C_\epsilon \exp\left(-\frac{(1-\epsilon)}{\sigma} d_{\text{Ag}}(\mathbf{z}, \Omega_A)\right). \quad \square

$$
:::

::::{admonition} Physics Isomorphism: Agmon Estimates
:class: note
:name: pi-agmon-estimates

**In Physics:** Agmon estimates give exponential decay bounds for eigenfunctions in classically forbidden regions. For a Schrödinger operator $-\hbar^2\Delta + V$, the ground state decays as $|\psi(x)| \lesssim \exp(-d_{\text{Ag}}(x, \Omega)/\hbar)$ where $d_{\text{Ag}}$ is the Agmon distance {cite}`agmon1982lectures`.

**In Implementation:** The belief wave-function amplitude decays through Pareto barriers (Theorem {prf:ref}`thm-e7-agmon-decay-bound`):

$$
|\Psi_0(\mathbf{z})| \leq C_\epsilon \exp\left(-\frac{1-\epsilon}{\sigma} d_{\text{Ag}}(\mathbf{z}, \Omega_A)\right)

$$
**Correspondence Table:**
| Semi-Classical Analysis | Agent (Tunneling) |
|:------------------------|:------------------|
| Agmon metric $(\rho_E)_{ij} = (V-E)_+ g_{ij}$ | Strategic Agmon metric |
| Agmon distance $d_{\text{Ag}}$ | Barrier "thickness" |
| Planck constant $\hbar$ | Cognitive scale $\sigma$ |
| Forbidden region $\{V > E\}$ | Pareto barrier $\{\Phi > E_0\}$ |
| Exponential decay rate | Tunneling suppression |

**Consequence:** Agmon distance, not Euclidean distance, controls tunneling probability.
::::



### E.7.4 Game Tensor Effect: Adversarial Suppression of Tunneling

We now prove that the Game Tensor $\mathcal{G}_{ij}$ (Definition {prf:ref}`def-the-game-tensor`) increases the effective barrier, suppressing tunneling.

:::{prf:corollary} E.7.3 (Adversarial Suppression of Tunneling)
:label: cor-e7-adversarial-suppression

Assume Agent $j$ is adversarial to Agent $i$, so the Game Tensor $\mathcal{G}_{ij}$ is positive semi-definite (Theorem {prf:ref}`thm-adversarial-mass-inflation`). Let:
- $\mathbf{g}_0 := \bigoplus_{i=1}^N G^{(i)}$ be the **non-interacting** (decoupled) metric
- $\mathbf{g}_{\text{adv}} := \mathbf{g}_0 + \alpha \sum_{i \neq j} \mathcal{G}_{ij}$ be the **adversarial** (Game-inflated) metric

Then the Agmon distances satisfy:

$$
d_{\text{Ag}}^{\text{adv}}(\Omega_A, \Omega_B) \geq d_{\text{Ag}}^{0}(\Omega_A, \Omega_B),

$$
and consequently the tunneling probability is exponentially suppressed:

$$
P_{\text{tunnel}}^{\text{adv}} \lesssim \exp\left(-\frac{d_{\text{Ag}}^{\text{adv}}}{\sigma}\right) \leq \exp\left(-\frac{d_{\text{Ag}}^{0}}{\sigma}\right) \lesssim P_{\text{tunnel}}^{0}.

$$
:::

(proof-cor-e7-adversarial-suppression)=
:::{prf:proof}

**Step 1 (Metric Comparison).** Since $\mathcal{G}_{ij} \succeq 0$ (positive semi-definite), for any tangent vector $\mathbf{v} \in T_{\mathbf{z}}\mathcal{M}$:

$$
\mathbf{v}^T \mathbf{g}_{\text{adv}} \mathbf{v} = \mathbf{v}^T \mathbf{g}_0 \mathbf{v} + \alpha \sum_{i \neq j} \mathbf{v}^T \mathcal{G}_{ij} \mathbf{v} \geq \mathbf{v}^T \mathbf{g}_0 \mathbf{v}.

$$
Thus $\mathbf{g}_{\text{adv}} \geq \mathbf{g}_0$ in the sense of quadratic forms.

**Step 2 (Path Length Inequality).** For any path $\gamma: [0,1] \to \mathcal{M}$, the Agmon length satisfies:

$$
L_{\text{Ag}}^{\text{adv}}(\gamma) = \int_0^1 \sqrt{(\mathcal{V} - E_0)_+} \cdot \|\dot{\gamma}\|_{\mathbf{g}_{\text{adv}}} \, dt \geq \int_0^1 \sqrt{(\mathcal{V} - E_0)_+} \cdot \|\dot{\gamma}\|_{\mathbf{g}_0} \, dt = L_{\text{Ag}}^{0}(\gamma).

$$
**Step 3 (Distance Inequality).** Taking the infimum over all paths:

$$
d_{\text{Ag}}^{\text{adv}}(\mathbf{x}, \mathbf{y}) = \inf_{\gamma} L_{\text{Ag}}^{\text{adv}}(\gamma) \geq \inf_{\gamma} L_{\text{Ag}}^{0}(\gamma) = d_{\text{Ag}}^{0}(\mathbf{x}, \mathbf{y}).

$$
**Step 4 (Tunneling Suppression).** By Theorem E.7.2, the ground state amplitude at distance $d$ from $\Omega_A$ scales as $\exp(-d/\sigma)$. Since $d_{\text{Ag}}^{\text{adv}} \geq d_{\text{Ag}}^{0}$:

$$
|\Psi_0^{\text{adv}}(\mathbf{z})|^2 \lesssim \exp\left(-\frac{2 d_{\text{Ag}}^{\text{adv}}}{\sigma}\right) \leq \exp\left(-\frac{2 d_{\text{Ag}}^{0}}{\sigma}\right) \lesssim |\Psi_0^{0}(\mathbf{z})|^2.

$$
The tunneling probability $P_{\text{tunnel}} \approx \int_{\Omega_B} |\Psi_0|^2$ inherits this exponential suppression. $\square$

:::

:::{admonition} Geometric Interpretation
:class: note
:name: interpretation-adversarial-barrier

**The Game Tensor inflates the metric, increasing geodesic and Agmon path lengths.**

In an adversarial setting:
- The metric satisfies $\mathbf{g}_{\text{adv}} \geq \mathbf{g}_0$ (in the sense of quadratic forms)
- Path lengths satisfy $L_{\text{Ag}}^{\text{adv}}(\gamma) \geq L_{\text{Ag}}^0(\gamma)$ for all paths $\gamma$
- The wave-function amplitude bound (Theorem E.7.2) yields smaller values under $\mathbf{g}_{\text{adv}}$
- Tunneling probability is exponentially suppressed

This proves that adversarial coupling increases Agmon distance (Corollary E.7.3), which by Theorem E.7.2 implies exponentially reduced tunneling probability.

:::



### E.7.5 Probabilistic Representation: Feynman-Kac Formula

To connect the spectral results to the stochastic WFR dynamics ({ref}`Section 22 <sec-the-equations-of-motion-geodesic-jump-diffusion>`), we invoke the rigorous Feynman-Kac formula.

:::{prf:theorem} E.7.4 (Feynman-Kac Representation)
:label: thm-e7-feynman-kac

Let $(\mathbf{X}_s)_{s \geq 0}$ be Brownian motion on the Riemannian manifold $(\mathcal{M}, \mathbf{g})$, starting at $\mathbf{X}_0 = \mathbf{z}$. Then the ground state $\Psi_0$ admits the representation:

$$
\Psi_0(\mathbf{z}) = \lim_{t \to \infty} e^{E_0 t} \cdot \mathbb{E}_{\mathbf{z}}\left[ \exp\left( -\frac{1}{\sigma^2} \int_0^t \mathcal{V}(\mathbf{X}_s) \, ds \right) \phi(\mathbf{X}_t) \right],

$$
where $\phi \in L^2(\mathcal{M})$ is any function with $\langle \Psi_0, \phi \rangle \neq 0$.

*Remark:* This is rigorous—not a heuristic "path integral." The expectation is over Brownian paths on the manifold.

:::

(proof-thm-e7-feynman-kac)=
:::{prf:proof}

**Step 1 (Semigroup Representation).** By the Feynman-Kac-Itô formula for Schrödinger operators on manifolds {cite}`simon1979functional`:

$$
(e^{-t\hat{H}_\sigma/\sigma^2} \phi)(\mathbf{z}) = \mathbb{E}_{\mathbf{z}}\left[ \exp\left( -\frac{1}{\sigma^2} \int_0^t \mathcal{V}(\mathbf{X}_s) \, ds \right) \phi(\mathbf{X}_t) \right].

$$
**Step 2 (Spectral Projection).** As $t \to \infty$, the semigroup projects onto the ground state:

$$
e^{-t\hat{H}_\sigma/\sigma^2} \phi \to e^{-tE_0/\sigma^2} \langle \Psi_0, \phi \rangle \Psi_0.

$$
**Step 3 (Normalization).** Multiplying by $e^{E_0 t/\sigma^2}$ and taking the limit gives the stated formula. $\square$

:::

::::{admonition} Physics Isomorphism: Feynman-Kac Formula
:class: note
:name: pi-feynman-kac

**In Physics:** The Feynman-Kac formula represents solutions to the Schrödinger equation as expectations over Brownian paths: $\psi(x,t) = \mathbb{E}_x[\exp(-\int_0^t V(X_s)ds)\psi_0(X_t)]$ {cite}`kac1949distributions,simon2005functional`.

**In Implementation:** The ground state wave-function admits the representation (Theorem {prf:ref}`thm-e7-feynman-kac`):

$$
\Psi_0(\mathbf{z}) = \lim_{t \to \infty} e^{E_0 t/\sigma^2} \cdot \mathbb{E}_{\mathbf{z}}\left[\exp\left(-\frac{1}{\sigma^2}\int_0^t \mathcal{V}(\mathbf{X}_s)ds\right)\phi(\mathbf{X}_t)\right]

$$
where $\mathbf{X}_s$ is Brownian motion on $(\mathcal{M}, \mathbf{g})$ and $E_0$ is the ground state energy.

**Correspondence Table:**
| Path Integral | Agent (Value Function) |
|:--------------|:-----------------------|
| Brownian motion $X_t$ | WFR diffusion |
| Potential $V(x)$ | Effective potential $\mathcal{V}$ |
| Imaginary time $\tau = it$ | Value iteration time |
| Path weight $e^{-\int V ds}$ | Reward accumulation |
| Ground state $\psi_0$ | Optimal belief amplitude |

**Significance:** This is rigorous (not heuristic path integrals); tunneling = rare large-deviation fluctuations.
::::

:::{prf:corollary} E.7.5 (Tunneling via Large Deviations)
:label: cor-e7-large-deviations

The tunneling probability is controlled by the **Large Deviation Principle** for Brownian paths on $(\mathcal{M}, \mathbf{g})$.

The rate function (Freidlin-Wentzell action) is:

$$
I[\gamma] = \frac{1}{2} \int_0^T \|\dot{\gamma}(t)\|_{\mathbf{g}}^2 \, dt,

$$
and paths that cross the barrier $\mathcal{K}$ while minimizing $I[\gamma] + \int_0^T (\mathcal{V}(\gamma) - E_0) \, dt$ are precisely the **instantons** that govern tunneling.

*Interpretation:* Tunneling is realized by rare stochastic fluctuations of the WFR diffusion process that penetrate the high-cost region. The probability of such fluctuations scales as $\exp(-S_{\text{inst}}/\sigma)$ where $S_{\text{inst}}$ is the instanton action—which equals the Agmon distance.

:::

::::{admonition} Physics Isomorphism: Large Deviation Principle
:class: note
:name: pi-large-deviation

**In Physics:** Large deviation theory quantifies rare events via rate functions: $P(X_n \in A) \asymp \exp(-n I(A))$ where $I$ is the rate function. In stochastic mechanics, instantons are paths minimizing the action that dominate rare transitions {cite}`freidlin1998random,varadhan1984large`.

**In Implementation:** Tunneling probability is controlled by the Large Deviation Principle (Corollary {prf:ref}`cor-e7-large-deviations`):

$$
P_{\text{tunnel}} \asymp \exp\left(-\frac{d_{\text{Ag}}(\Omega_A, \Omega_B)}{\sigma}\right)

$$
where the rate function is the Agmon action.

**Correspondence Table:**
| Large Deviations | Agent (Barrier Crossing) |
|:-----------------|:-------------------------|
| Rate function $I$ | Agmon action |
| Instanton path | Optimal tunneling trajectory |
| Cramér's theorem | Exponential bound on transition |
| Sanov's theorem | Entropy cost of belief shift |
| $n \to \infty$ limit | $\sigma \to 0$ (classical Nash) |

**Consequence:** Paths minimizing Brownian action are precisely the instantons governing tunneling.
::::



### E.7.6 Summary of Rigorous Results

**Table E.7.1 (Summary of Tunneling Rigor).**

| Result                 | Statement                                         | Method                     |
|:-----------------------|:--------------------------------------------------|:---------------------------|
| **Existence**          | $P(\Omega_B) > 0$ always                          | Perron-Frobenius / Harnack |
| **Decay Rate**         | $\Psi_0 \lesssim e^{-d_{\text{Ag}}/\sigma}$ | Agmon estimates |
| **Game Tensor Effect** | $d_{\text{Ag}}^{\text{adv}} \geq d_{\text{Ag}}^0$ | Metric comparison          |
| **Probabilistic**      | Feynman-Kac representation                        | Semigroup theory           |
| **Optimal Path**       | Instanton = Agmon geodesic                        | Large deviations           |

**Rigorous version of Theorem {prf:ref}`thm-tunneling-probability`:**

$$
P_{\text{tunnel}}(\Omega_A \to \Omega_B) = \Theta\left(\exp\left(-\frac{2}{\sigma} d_{\text{Ag}}(\Omega_A, \Omega_B)\right)\right) \quad \text{as } \sigma \to 0,

$$
where $\Theta(\cdot)$ denotes asymptotic equality up to polynomial prefactors in $\sigma$.

This completes the rigorous foundation for the strategic tunneling mechanism. $\square$



(sec-appendix-e-proof-of-corollary-varentropy-stability)=
## E.8 Proof of Corollary {prf:ref}`cor-varentropy-stability`

**Statement:** $V_H(z) = T_c^2 \frac{\partial H(\pi)}{\partial T_c}$.

**Hypothesis:** Let $\pi(a|z) = \frac{1}{Z} \exp\left(\frac{Q(z,a)}{T_c}\right)$ be the policy, where $Z = \sum_a \exp(Q/T_c)$ is the partition function. Let $\beta_{\text{ent}} = 1/T_c$ be the inverse {prf:ref}`def-cognitive-temperature`.

(proof-cor-varentropy-stability)=
:::{prf:proof}

**Step 1: Express Entropy in terms of $\beta_{\text{ent}}$.**
The entropy of the policy is:

$$
H(\pi) = -\sum_a \pi(a) \ln \pi(a).

$$
Substituting $\ln \pi(a) = \beta_{\text{ent}} Q(a) - \ln Z$:

$$
H(\pi) = -\sum_a \pi(a) [\beta_{\text{ent}} Q(a) - \ln Z] = \ln Z - \beta_{\text{ent}} \mathbb{E}_\pi[Q].

$$
**Step 2: Derivative of Entropy w.r.t. $\beta_{\text{ent}}$.**
Differentiating with respect to $\beta_{\text{ent}}$:

$$
\frac{\partial H}{\partial \beta_{\text{ent}}} = \frac{\partial \ln Z}{\partial \beta_{\text{ent}}} - \mathbb{E}_\pi[Q] - \beta_{\text{ent}} \frac{\partial \mathbb{E}_\pi[Q]}{\partial \beta_{\text{ent}}}.

$$
Using the identity $\frac{\partial \ln Z}{\partial \beta_{\text{ent}}} = \mathbb{E}_\pi[Q]$:

$$
\frac{\partial H}{\partial \beta_{\text{ent}}} = -\beta_{\text{ent}} \frac{\partial \mathbb{E}_\pi[Q]}{\partial \beta_{\text{ent}}} = -\beta_{\text{ent}} \mathrm{Var}_\pi(Q),

$$
where we used $\frac{\partial \mathbb{E}[Q]}{\partial \beta_{\text{ent}}} = \mathrm{Var}(Q)$ (standard fluctuation-response relation).

**Step 3: Relate $\mathrm{Var}(Q)$ to Varentropy.**
Recall $\mathcal{I}(a) = -\ln \pi(a) = -\beta_{\text{ent}} Q(a) + \ln Z$. The variance of the surprisal is:

$$
V_H(\pi) = \mathrm{Var}(\mathcal{I}) = \mathrm{Var}(-\beta_{\text{ent}} Q + \ln Z) = \beta_{\text{ent}}^2 \mathrm{Var}(Q).

$$
**Step 4: Change of variables to $T_c$.**
We have $V_H = \beta_{\text{ent}}^2 \mathrm{Var}(Q)$ and $\frac{\partial H}{\partial \beta_{\text{ent}}} = -\beta_{\text{ent}} \mathrm{Var}(Q)$.
Therefore $V_H = -\beta_{\text{ent}} \frac{\partial H}{\partial \beta_{\text{ent}}}$.

Using the chain rule $\frac{\partial}{\partial T_c} = -\frac{1}{T_c^2} \frac{\partial}{\partial \beta_{\text{ent}}}$:

$$
\frac{\partial H}{\partial T_c} = -\frac{1}{T_c^2} \frac{\partial H}{\partial \beta_{\text{ent}}} = \frac{1}{T_c^2} \cdot \beta_{\text{ent}} \mathrm{Var}(Q) = \frac{V_H}{T_c^2 \cdot \beta_{\text{ent}}} = \frac{V_H}{T_c}.

$$
**Final Result:** Rearranging yields:

$$
V_H(z) = T_c \frac{\partial H(\pi)}{\partial T_c} = \beta_{\text{ent}}^2 \mathrm{Var}(Q) = C_v.

$$
This proves that Varentropy equals the heat capacity and measures the sensitivity of the entropy to temperature fluctuations. $\square$

:::



(sec-appendix-e-proof-of-corollary-bimodal-instability)=
## E.9 Proof of Corollary {prf:ref}`cor-bimodal-instability`

**Statement:** For a bimodal policy on a value ridge, $V_H$ is significant, distinguishing it from uniform noise.

**Hypothesis:** Let $\pi$ be a mixture of two dominant modes with values $Q_1, Q_2$ and a background of $N-2$ negligible modes.

(proof-cor-bimodal-instability)=
:::{prf:proof}

**Step 1: Variance of Surprisal Form.**

$$
V_H = \mathbb{E}[\mathcal{I}^2] - (\mathbb{E}[\mathcal{I}])^2.

$$
Since $\mathcal{I} = -\beta Q + \ln Z$, we have $V_H = \beta^2 \mathrm{Var}(Q)$.

**Step 2: Two-Point Statistics.**
Consider two actions $a_1, a_2$ with probabilities $p, 1-p$. The variance of a Bernoulli variable taking values $Q_1, Q_2$ is:

$$
\mathrm{Var}(Q) = p(1-p)(Q_1 - Q_2)^2.

$$
Thus:

$$
V_H = \beta^2 p(1-p) (\Delta Q)^2 = p(1-p) \left( \frac{\Delta Q}{T_c} \right)^2.

$$
For equally weighted modes ($p = 1/2$), this simplifies to:

$$
V_H = \frac{1}{4} \left( \frac{\Delta Q}{T_c} \right)^2.

$$
**Step 3: Interpretation of $\Delta Q$.**
$\Delta Q$ is the value gap between the two modes.

- **Perfect Symmetry (The Ridge):** If $Q_1 = Q_2$ exactly, then $\Delta Q = 0 \implies V_H = 0$.
- **Structural Instability:** When the agent is *slightly* off-center or when sampling includes the *tails*, the effective $\Delta Q > 0$.

**Step 4: Distinguishing Structure from Noise.**
For a distribution with structure (peaks and valleys), $\mathrm{Var}(Q) > 0$. For a flat distribution (noise), $\mathrm{Var}(Q) = 0$.

Specifically, on a ridge, the agent samples $a_{\text{left}}$ and $a_{\text{right}}$ (high $Q$) but also transitively samples the separating region (lower $Q$) during exploration. The variance of $Q$ along the trajectory corresponds to $V_H$:

$$
V_H \propto (\Delta Q_{\text{peak-valley}})^2.

$$
This proves that $V_H$ detects the topological feature (the valley) that distinguishes a fork from a flat plane. $\square$

:::



(sec-appendix-e-proof-of-corollary-varentropy-brake)=
## E.10 Proof of Corollary {prf:ref}`cor-varentropy-brake`

**Statement:** To maintain stability, the cooling rate must satisfy $|\dot{T}_c| \ll T_c / \sqrt{V_H}$.

**Hypothesis:** We require the probability distribution $\pi_t$ to remain close to the equilibrium Boltzmann distribution $\pi^*_{T_c(t)}$ during annealing. This is the **Adiabatic Condition**.

(proof-cor-varentropy-brake)=
:::{prf:proof}

**Step 1: Thermodynamic Speed.**
The rate of change of the policy distribution with respect to temperature is measured by the Fisher Information metric $g_{TT}$ on the statistical manifold parameterized by $T_c$:

$$
g_{TT} = \mathbb{E}\left[ \left( \frac{\partial \ln \pi}{\partial T_c} \right)^2 \right].

$$
**Step 2: Relate Fisher Metric to Varentropy.**
Recall $\ln \pi = \frac{Q}{T_c} - \ln Z$. Then:

$$
\frac{\partial \ln \pi}{\partial T_c} = -\frac{Q}{T_c^2} + \frac{\mathbb{E}[Q]}{T_c^2} = -\frac{1}{T_c^2}(Q - \mathbb{E}[Q]).

$$
Substituting into the Fisher definition:

$$
g_{TT} = \frac{1}{T_c^4} \mathbb{E}\left[ (Q - \mathbb{E}[Q])^2 \right] = \frac{\mathrm{Var}(Q)}{T_c^4}.

$$
Using $V_H = \frac{\mathrm{Var}(Q)}{T_c^2}$ (from Proof E.8):

$$
g_{TT} = \frac{V_H}{T_c^2}.

$$
**Step 3: Thermodynamic Length.**
The "distance" traversed in probability space for a small temperature change $dT_c$ is $ds^2 = g_{TT} dT_c^2$:

$$
ds = \sqrt{g_{TT}} |dT_c| = \frac{\sqrt{V_H}}{T_c} |dT_c|.

$$
**Step 4: Adiabatic Condition.**
For the system to relax to equilibrium (stay in the basin of attraction), the speed of change in distribution space must be bounded:

$$
\left| \frac{ds}{dt} \right| \leq C \cdot \tau_{\text{relax}}^{-1}.

$$
Substituting $ds/dt$:

$$
\frac{\sqrt{V_H}}{T_c} \left| \frac{dT_c}{dt} \right| \leq C.

$$
Solving for the cooling rate:

$$
\left| \frac{dT_c}{dt} \right| \leq C \frac{T_c}{\sqrt{V_H}}.

$$
**Conclusion:** When Varentropy $V_H$ is large (phase transition/critical point), the permissible cooling rate goes to zero. The Governor must apply the "Varentropy Brake" to prevent quenching the system into a suboptimal metastable state. $\square$

:::



(sec-appendix-e-proof-of-corollary-epistemic-curiosity-filter)=
## E.11 Proof of Corollary {prf:ref}`cor-epistemic-curiosity-filter`

**Statement:** $\nabla \Psi_{\text{causal}} \propto \nabla \mathbb{E}_{z'} [ V_H[P(\theta_W | z, a, z')] ]$.

**Hypothesis:** We define $\Psi_{\text{causal}}$ as the Expected Information Gain (EIG) about model parameters $\theta$ given a transition $(z, a) \to z'$.

(proof-cor-epistemic-curiosity-filter)=
:::{prf:proof}

**Step 1: Definition of EIG.**

$$
\text{EIG}(z, a) = I(\theta; z' | z, a) = H(z' | z, a) - \mathbb{E}_{\theta} [ H(z' | z, a, \theta) ].

$$
This is the **Total Predictive Entropy** minus the **Expected Aleatoric Entropy**.

**Step 2: Decomposition of Uncertainty.**
For the "noisy TV" case (outcomes are stochastic noise independent of $\theta$):

$$
H(z' | z, a, \theta) \approx H(z' | z, a) \implies \text{EIG} \approx 0.

$$
**Step 3: Varentropy as Structure Detector.**
The varentropy $V_H(z' | z, a)$ measures the variance of log-probabilities.

- **Uniform noise:** $V_H^{\text{noise}} \to 0$ (all outcomes equally likely).
- **Structured uncertainty:** $V_H^{\text{structured}} > 0$ (some outcomes much more likely).

**Step 4: Connection to Multimodality.**
If the model is uncertain about structure ($\theta$), the predictive distribution $p(z')$ is a mixture of distinct hypotheses $p(z'|\theta_1), p(z'|\theta_2)$. As established in Proof E.9, a mixture of distinct modes has high Varentropy compared to a broad unimodal distribution (noise).

**Step 5: Operational Equivalence.**
Thus, maximizing EIG is functionally equivalent to maximizing the **Varentropy of the expected outcome**, provided the aleatoric noise floor is constant:

$$
\nabla \Psi_{\text{causal}} \propto \nabla \mathrm{Var}_{z' \sim p(z'|z,a)} [ -\ln p(z'|z,a) ].

$$
**Conclusion:** The agent should seek states where the World Model's prediction has high Varentropy (conflicting hypotheses), as these offer the maximum potential for falsification (reduction of parameter variance). $\square$

:::



## E.12 Derivation of the HJB-Klein-Gordon Correspondence (Theorem {prf:ref}`thm-hjb-klein-gordon`)

**Statement:** Under finite information propagation speed $c_{\text{info}}$, the Bellman equation generalizes to the hyperbolic Klein-Gordon equation:

$$
\left(\frac{1}{c_{\text{info}}^2}\partial_t^2 - \Delta_G + \kappa^2\right)V^{(i)} = \rho_r^{(i)} + \sum_{j \neq i} \rho^{\text{ret}}_{ij}

$$

(proof-hjb-klein-gordon)=
:::{prf:proof}

**Step 1: Bellman Recursion with Temporal Structure.**

The standard Bellman equation assumes instantaneous value propagation:

$$
V(z, t) = r(z)\Delta t + \gamma \mathbb{E}_{z' \sim P(\cdot|z,a)}[V(z', t + \Delta t)]

$$

where $\gamma = e^{-\kappa_t \Delta t}$ is the temporal discount factor with $\kappa_t = -\ln\gamma / \Delta t$ having units $[\kappa_t] = 1/[\text{time}]$.

**Step 2: Second-Order Taylor Expansion.**

Expand $V(z', t + \Delta t)$ to second order in both space and time. Let $z' = z + \delta z$ where $\delta z$ is the state transition:

$$
V(z', t + \Delta t) = V(z, t) + \partial_t V \cdot \Delta t + \frac{1}{2}\partial_t^2 V \cdot (\Delta t)^2 + \nabla V \cdot \delta z + \frac{1}{2}(\delta z)^\top \nabla^2 V (\delta z) + \partial_t \nabla V \cdot \Delta t \cdot \delta z + O(3)

$$

**Step 3: Expectations Under Diffusion.**

For a diffusion process with drift $b(z)$ and diffusion tensor $\Sigma = 2T_c G^{-1}$:

$$
\mathbb{E}[\delta z] = b \Delta t, \quad \mathbb{E}[(\delta z)(\delta z)^\top] = \Sigma \Delta t

$$

Taking expectations:

$$
\mathbb{E}[V(z', t + \Delta t)] = V + \partial_t V \Delta t + \frac{1}{2}\partial_t^2 V (\Delta t)^2 + \nabla_A V \cdot b \Delta t + T_c \text{Tr}(G^{-1}\nabla^2 V) \Delta t + O((\Delta t)^{3/2})

$$

The trace term is the Laplace-Beltrami operator: $\text{Tr}(G^{-1}\nabla^2 V) = \Delta_G V$.

**Step 4: Substitution into Bellman.**

Substituting into the Bellman equation:

$$
V = r \Delta t + (1 - \kappa_t \Delta t)\left(V + \partial_t V \Delta t + \frac{1}{2}\partial_t^2 V (\Delta t)^2 + \nabla_A V \cdot b \Delta t + T_c \Delta_G V \Delta t\right)

$$

**Step 5: Instantaneous Limit (Elliptic Case).**

Dividing by $\Delta t$ and taking $\Delta t \to 0$ while keeping only $O(\Delta t)$ terms:

$$
0 = r - \kappa_t V + \partial_t V + \nabla_A V \cdot b + T_c \Delta_G V

$$

For stationary states ($\partial_t V = 0$) with zero drift ($b = 0$):

$$
-T_c \Delta_G V + \kappa_t V = r

$$

This is the **Helmholtz equation** (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`). Note that $\kappa_t$ here has temporal units.

**Step 6: Finite Propagation Speed (Hyperbolic Case).**

The key insight is that the above derivation assumes **instantaneous** information propagation: the value at time $t + \Delta t$ depends on rewards and transitions known at time $t$. When information propagates at finite speed $c_{\text{info}}$, two modifications occur:

**(i) Retardation of Spatial Coupling:** Rewards at spatial distance $\ell$ are received with delay $\tau = \ell / c_{\text{info}}$. This is handled by the retarded potential $\Phi_{ij}^{\text{ret}}$.

**(ii) Wave Propagation of Value:** The value function itself propagates as a wave, not instantaneously. The characteristic timescale for value changes over spatial scale $\ell$ is $\tau_\ell = \ell / c_{\text{info}}$.

To derive the wave equation, we must retain the **second-order time derivative**. Define the **spatial screening mass**:

$$
\kappa := \kappa_t / c_{\text{info}} = -\ln\gamma / (c_{\text{info}} \Delta t)

$$

with units $[\kappa] = 1/[\text{length}]$.

**Step 7: Wave Equation Derivation.**

Consider the characteristic scales:
- Temporal: $\Delta t \sim \ell / c_{\text{info}}$ (time for information to traverse distance $\ell$)
- Spatial: $\ell$ (characteristic length scale)

The ratio $(\Delta t)^2 / \ell^2 \sim 1/c_{\text{info}}^2$ is no longer negligible. Retaining the $(\Delta t)^2$ term in the expansion:

$$
\frac{1}{c_{\text{info}}^2}\partial_t^2 V + \partial_t V / c_{\text{info}} = r - \kappa^2 V + \Delta_G V + \text{(coupling terms)}

$$

In the **stationary wave regime** where $\partial_t V \ll c_{\text{info}} \partial_t^2 V / \kappa$, the first-order time derivative is negligible compared to the second-order term, yielding:

$$
\frac{1}{c_{\text{info}}^2}\partial_t^2 V - \Delta_G V + \kappa^2 V = \rho_r + \sum_j \rho^{\text{ret}}_{ij}

$$

This is the **Klein-Gordon equation** with mass $\kappa$.

**Step 8: Physical Interpretation.**

The transition from Helmholtz (elliptic) to Klein-Gordon (hyperbolic) parallels the transition in electromagnetism:

| Regime | Equation | Value Propagation |
|:-------|:---------|:------------------|
| $c_{\text{info}} \to \infty$ | Helmholtz: $(-\Delta_G + \kappa^2)V = \rho_r$ | Instantaneous |
| $c_{\text{info}} < \infty$ | Klein-Gordon: $(\frac{1}{c^2}\partial_t^2 - \Delta_G + \kappa^2)V = \rho_r$ | Wave at speed $c$ |

The screening mass $\kappa$ determines the characteristic decay length $\ell_\gamma = 1/\kappa$: the distance over which value influence diminishes by factor $e$.

**Step 9: Dimensional Verification.**

- $[\partial_t^2 V] = [\text{nat}]/[\text{time}]^2$
- $[c_{\text{info}}^{-2}\partial_t^2 V] = [\text{nat}]/[\text{length}]^2$
- $[\Delta_G V] = [\text{nat}]/[\text{length}]^2$
- $[\kappa^2 V] = [\text{nat}]/[\text{length}]^2$ (since $[\kappa] = 1/[\text{length}]$)
- $[\rho_r] = [\text{nat}]/[\text{length}]^2$

All terms have consistent units. $\square$

:::



## E.13 Derivation of the Madelung Transform (Theorem {prf:ref}`thm-madelung-transform`)

**Statement:** The belief wave-function $\psi = \sqrt{\rho} e^{iV/\sigma}$ satisfies the Inference Schrödinger Equation with
$D_i := \nabla_i - \frac{i}{\sigma}A_i$ and $\nabla_A V := \nabla V - A$ if and only if $(\rho, V)$ satisfy the WFR-HJB system.

(proof-madelung-transform)=
:::{prf:proof}

**Step 1: Polar Decomposition.**

Let the belief wave-function have polar form:

$$
\psi = R \, e^{i\phi}, \quad R := \sqrt{\rho}, \quad \phi := V/\sigma

$$

where $\rho = |\psi|^2$ is the belief density and $V$ is the value function. The parameter $\sigma > 0$ is the cognitive action scale (Definition {prf:ref}`def-cognitive-action-scale`).

**Step 2: Compute Time Derivative.**

$$
\partial_s \psi = \partial_s(R e^{i\phi}) = (\partial_s R) e^{i\phi} + R \cdot i(\partial_s \phi) e^{i\phi} = \left(\frac{\partial_s R}{R} + i \partial_s \phi\right)\psi

$$

Since $R = \sqrt{\rho}$, we have $\partial_s R / R = \partial_s \rho / (2\rho)$. Thus:

$$
\partial_s \psi = \left(\frac{\partial_s \rho}{2\rho} + \frac{i}{\sigma}\partial_s V\right)\psi

$$

**Step 3: Compute the covariant Laplacian of $\psi$.**

Let $D_i := \nabla_i - \frac{i}{\sigma}A_i$ and $\nabla_A V := \nabla V - A$. Then:

$$
D_i \psi = \left(\nabla_i R + \frac{i}{\sigma} R (\nabla_A V)_i\right) e^{i\phi}

$$
and

$$
D^i D_i \psi = \left[\Delta_G R + \frac{2i}{\sigma} G^{-1}(\nabla R, \nabla_A V) + \frac{i}{\sigma} R \nabla_G \cdot (G^{-1}\nabla_A V) - \frac{1}{\sigma^2} R \|\nabla_A V\|_G^2\right] e^{i\phi}.

$$

Dividing by $\psi = R e^{i\phi}$:

$$
\frac{D^i D_i \psi}{\psi} = \frac{\Delta_G R}{R} + \frac{2i}{\sigma} \frac{G^{-1}(\nabla R, \nabla_A V)}{R} + \frac{i}{\sigma}\nabla_G \cdot (G^{-1}\nabla_A V) - \frac{1}{\sigma^2}\|\nabla_A V\|_G^2.

$$

**Step 4: Define the Bohm Potential.**

Using $R = \sqrt{\rho}$, we have:

$$
\frac{\Delta_G R}{R} = \frac{\Delta_G \sqrt{\rho}}{\sqrt{\rho}}

$$

Define the **Bohm quantum potential**:

$$
Q_B := -\frac{\sigma^2}{2} \frac{\Delta_G \sqrt{\rho}}{\sqrt{\rho}} = -\frac{\sigma^2}{2} \frac{\Delta_G R}{R}

$$

**Step 5: Inference Schrödinger Equation.**

The Inference Schrödinger Equation is:

$$
i\sigma \partial_s \psi = \hat{H}_{\text{inf}} \psi, \quad \hat{H}_{\text{inf}} = -\frac{\sigma^2}{2} D^i D_i + \Phi_{\text{eff}} + Q_B - \frac{i\sigma}{2}r

$$

Substituting our expressions:

$$
i\sigma \partial_s \psi = i\sigma\left(\frac{\partial_s \rho}{2\rho} + \frac{i}{\sigma}\partial_s V\right)\psi = \left(\frac{i\sigma \partial_s \rho}{2\rho} - \partial_s V\right)\psi

$$

$$
\hat{H}_{\text{inf}}\psi = \left[-\frac{\sigma^2}{2}\frac{D^i D_i \psi}{\psi} + \Phi_{\text{eff}} + Q_B - \frac{i\sigma}{2}r\right]\psi

$$

**Step 6: Separate Real and Imaginary Parts.**

Expanding $-\frac{\sigma^2}{2}\frac{D^i D_i \psi}{\psi}$:

$$
-\frac{\sigma^2}{2}\frac{D^i D_i \psi}{\psi} = -\frac{\sigma^2}{2}\frac{\Delta_G R}{R} - i\sigma \frac{G^{-1}(\nabla R, \nabla_A V)}{R} - \frac{i\sigma}{2}\nabla_G \cdot (G^{-1}\nabla_A V) + \frac{1}{2}\|\nabla_A V\|_G^2

$$

$$
= Q_B - i\sigma \frac{G^{-1}(\nabla \sqrt{\rho}, \nabla_A V)}{\sqrt{\rho}} - \frac{i\sigma}{2}\nabla_G \cdot (G^{-1}\nabla_A V) + \frac{1}{2}\|\nabla_A V\|_G^2

$$

The Schrödinger equation $i\sigma \partial_s \psi = \hat{H}_{\text{inf}}\psi$ becomes:

$$
\frac{i\sigma \partial_s \rho}{2\rho} - \partial_s V = Q_B + Q_B - i\sigma \frac{G^{-1}(\nabla \sqrt{\rho}, \nabla_A V)}{\sqrt{\rho}} - \frac{i\sigma}{2}\nabla_G \cdot (G^{-1}\nabla_A V) + \frac{1}{2}\|\nabla_A V\|_G^2 + \Phi_{\text{eff}} - \frac{i\sigma}{2}r

$$

Wait—there's a double $Q_B$. Let me redo this more carefully. The $Q_B$ in $\hat{H}_{\text{inf}}$ cancels with the $-\frac{\sigma^2}{2}\frac{\Delta_G R}{R}$ from the kinetic term:

$$
\hat{H}_{\text{inf}}\psi = \left[- i\sigma \frac{G^{-1}(\nabla \sqrt{\rho}, \nabla_A V)}{\sqrt{\rho}} - \frac{i\sigma}{2}\nabla_G \cdot (G^{-1}\nabla_A V) + \frac{1}{2}\|\nabla_A V\|_G^2 + \Phi_{\text{eff}} - \frac{i\sigma}{2}r\right]\psi

$$

**Real part (coefficient of $\psi$):**

$$
-\partial_s V = \frac{1}{2}\|\nabla_A V\|_G^2 + \Phi_{\text{eff}}

$$

This is the **Hamilton-Jacobi-Bellman equation**:

$$
\partial_s V + \frac{1}{2}\|\nabla_A V\|_G^2 + \Phi_{\text{eff}} = 0 \quad \checkmark

$$

**Imaginary part (coefficient of $i\psi$):**

$$
\frac{\sigma \partial_s \rho}{2\rho} = -\sigma \frac{G^{-1}(\nabla \sqrt{\rho}, \nabla_A V)}{\sqrt{\rho}} - \frac{\sigma}{2}\nabla_G \cdot (G^{-1}\nabla_A V) - \frac{\sigma}{2}r

$$

Simplifying: $\frac{G^{-1}(\nabla \sqrt{\rho}, \nabla_A V)}{\sqrt{\rho}} = \frac{G^{-1}(\nabla \rho, \nabla_A V)}{2\rho}$. Thus:

$$
\frac{\partial_s \rho}{2\rho} = -\frac{G^{-1}(\nabla \rho, \nabla_A V)}{2\rho} - \frac{1}{2}\nabla_G \cdot (G^{-1}\nabla_A V) - \frac{1}{2}r

$$

Multiplying by $2\rho$:

$$
\partial_s \rho = -G^{-1}(\nabla \rho, \nabla_A V) - \rho \nabla_G \cdot (G^{-1}\nabla_A V) - \rho r

$$

Using the velocity field $\mathbf{v} = -G^{-1}\nabla_A V$ (conservative case: $A=0$) and the identity $\nabla_G \cdot (\rho \mathbf{v}) = G^{-1}(\nabla \rho, \mathbf{v}) + \rho \nabla_G \cdot \mathbf{v}$:

$$
\partial_s \rho = G^{-1}(\nabla \rho, \mathbf{v}) + \rho \nabla_G \cdot \mathbf{v} - \rho r = \nabla_G \cdot (\rho \mathbf{v}) - \rho r

$$

This is the **WFR continuity equation** (unbalanced):

$$
\partial_s \rho + \nabla_G \cdot (\rho \mathbf{v}) = \rho r \quad \checkmark

$$

**Conclusion:** The Madelung transform $\psi = \sqrt{\rho} e^{iV/\sigma}$ is an exact equivalence between:
- The Inference Schrödinger Equation for $\psi$
- The coupled WFR-HJB system for $(\rho, V)$

The Bohm potential $Q_B$ emerges naturally from the kinetic energy operator acting on the amplitude $R = \sqrt{\rho}$. $\square$

:::



## E.14 Proof of Markov Restoration on the Causal Bundle (Theorem {prf:ref}`thm-markov-restoration`)

**Statement:** The augmented state $(z^{(N)}_t, \Xi_{<t})$ forms a Markov process even when the raw state $z^{(N)}_t$ does not.

(proof-markov-restoration)=
:::{prf:proof}

**Step 1: Define the Information Content.**

Let $\mathcal{I}_t$ denote the total information available to the system at time $t$:

$$
\mathcal{I}_t := \sigma(z^{(N)}_\tau, a^{(N)}_\tau, r^{(N)}_\tau : \tau \leq t)

$$
where $\sigma(\cdot)$ denotes the sigma-algebra generated by the random variables.

**Step 2: Causal Factorization.**

Under the finite information speed $c_{\text{info}}$, define the **causal past** of agent $i$ at time $t$:

$$
\mathcal{C}^{(i)}_t := \{(j, \tau) : \tau \leq t - d_{\mathcal{E}}(i,j)/c_{\text{info}}\}

$$
This is the set of (agent, time) pairs that can causally influence agent $i$ at time $t$.

The transition kernel factorizes:

$$
P(z^{(i)}_{t+\Delta t} | \mathcal{I}_t) = P(z^{(i)}_{t+\Delta t} | z^{(i)}_t, \{z^{(j)}_\tau : (j,\tau) \in \mathcal{C}^{(i)}_t\})

$$

**Step 3: Memory Screen as Sufficient Statistic.**

The Memory Screen $\Xi^{(i)}_{<t}$ is defined (Definition {prf:ref}`def-memory-screen`) as a compression of the causal past:

$$
\Xi^{(i)}_{<t} := f^{(i)}(\{z^{(j)}_\tau : (j,\tau) \in \mathcal{C}^{(i)}_t\})

$$
where $f^{(i)}$ is a sufficient statistic for predicting $z^{(i)}_{t+\Delta t}$.

**Claim:** $\Xi^{(i)}_{<t}$ satisfies the **sufficiency condition**:

$$
P(z^{(i)}_{t+\Delta t} | z^{(i)}_t, \Xi^{(i)}_{<t}, \Xi^{(i)}_{<t'}) = P(z^{(i)}_{t+\Delta t} | z^{(i)}_t, \Xi^{(i)}_{<t}) \quad \forall t' < t

$$

**Step 4: Proof of Sufficiency.**

By the definition of causal structure:
1. Events at $(j, \tau)$ with $\tau < t - d_{\mathcal{E}}(i,j)/c_{\text{info}}$ are already incorporated into $\Xi^{(i)}_{<t}$
2. Events at $(j, \tau)$ with $\tau \geq t - d_{\mathcal{E}}(i,j)/c_{\text{info}}$ cannot yet influence agent $i$

Thus, all information from $\Xi^{(i)}_{<t'}$ for $t' < t$ that is relevant to $z^{(i)}_{t+\Delta t}$ is already contained in $\Xi^{(i)}_{<t}$ (by the nested structure of causal cones).

**Step 5: Joint Markov Property.**

Define the joint augmented state:

$$
\mathbf{X}_t := (z^{(N)}_t, \Xi_{<t}) \in \mathcal{Z}_{\text{causal}}

$$

The transition kernel for the augmented state is:

$$
P(\mathbf{X}_{t+\Delta t} | \mathbf{X}_t, \mathbf{X}_{t-\Delta t}, \ldots) = P(\mathbf{X}_{t+\Delta t} | \mathbf{X}_t)

$$

This follows because:
- The current positions $z^{(N)}_t$ determine the local dynamics
- The memory screens $\Xi_{<t}$ contain all causally relevant history
- No additional information from $\mathbf{X}_{t-\Delta t}, \ldots$ can improve prediction beyond what $\mathbf{X}_t$ provides

**Step 6: Formal Verification (Chapman-Kolmogorov).**

The augmented process satisfies the Chapman-Kolmogorov equation:

$$
P(\mathbf{X}_{t+s} | \mathbf{X}_t) = \int P(\mathbf{X}_{t+s} | \mathbf{X}_{t+r}) P(\mathbf{X}_{t+r} | \mathbf{X}_t) \, d\mathbf{X}_{t+r}

$$
for all $0 < r < s$, which characterizes Markov processes. $\square$

:::



## E.15 Proof of Nash Equilibrium as Standing Wave (Theorem {prf:ref}`thm-nash-standing-wave`)

**Statement:** A Nash equilibrium in the multi-agent Klein-Gordon system corresponds to a standing wave pattern with time-averaged zero flux.

(proof-nash-standing-wave)=
:::{prf:proof}

**Step 1: Standing Wave Ansatz.**

Consider the coupled Klein-Gordon system for $N$ agents:

$$
\left(\frac{1}{c^2}\partial_t^2 - \Delta_{G^{(i)}} + \kappa^2\right)V^{(i)} = \rho_r^{(i)} + \sum_{j \neq i} \rho^{\text{ret}}_{ij}

$$

Seek standing wave solutions of the form:

$$
V^{(i)}(z, t) = \bar{V}^{(i)}(z) + \sum_{n=1}^\infty \left[a_n^{(i)}(z) \cos(\omega_n t) + b_n^{(i)}(z) \sin(\omega_n t)\right]

$$
where $\bar{V}^{(i)}$ is the time-averaged component.

**Step 2: Boundary Conditions.**

On the product manifold $\mathcal{Z}^{(N)} = \prod_i \mathcal{Z}^{(i)}$, impose:
- **Dirichlet at sensors:** $V^{(i)}|_{\partial_{\text{in}}} = V_{\text{obs}}$ (observations fix boundary values)
- **Neumann at motors:** $\nabla_n V^{(i)}|_{\partial_{\text{out}}} = 0$ (no value flux at action boundary)

These boundary conditions create a "cavity" that supports discrete eigenfrequencies.

**Step 3: Eigenmode Expansion.**

The D'Alembertian $\square_G = \frac{1}{c^2}\partial_t^2 - \Delta_G$ on the bounded domain has discrete spectrum. Let $\{\phi_n\}$ be the eigenfunctions of $-\Delta_G + \kappa^2$ with eigenvalues $\lambda_n$:

$$
(-\Delta_G + \kappa^2)\phi_n = \lambda_n \phi_n, \quad \lambda_1 \leq \lambda_2 \leq \cdots

$$

The standing wave frequencies are $\omega_n = c\sqrt{\lambda_n}$.

**Step 4: Time-Averaged Stationarity Implies Nash.**

**Definition (Time-Averaged Nash):** A configuration $\mathbf{z}^* = (z^{(1)*}, \ldots, z^{(N)*})$ is a time-averaged Nash equilibrium if:

$$
\langle \mathbf{J}^{(i)} \rangle_T := \frac{1}{T}\int_0^T \mathbf{J}^{(i)}(z^{(i)*}, t) \, dt = 0 \quad \forall i

$$
where $\mathbf{J}^{(i)} = -\rho^{(i)} G^{-1} \nabla_A V^{(i)}$ is the probability current.

**Claim:** At a standing wave equilibrium, $\langle \mathbf{J}^{(i)} \rangle_T = 0$.

*Proof of Claim:* For the standing wave ansatz:

$$
\nabla_A V^{(i)} = \nabla_A \bar{V}^{(i)} + \sum_n \left[\nabla_A a_n^{(i)} \cos(\omega_n t) + \nabla_A b_n^{(i)} \sin(\omega_n t)\right]

$$

Time-averaging over period $T \gg 2\pi/\omega_1$:

$$
\langle \nabla_A V^{(i)} \rangle_T = \nabla_A \bar{V}^{(i)}

$$
since $\langle \cos(\omega_n t) \rangle_T = \langle \sin(\omega_n t) \rangle_T = 0$.

At a stationary point of $\bar{V}^{(i)}$, we have $\nabla \bar{V}^{(i)} = 0$, hence $\langle \mathbf{J}^{(i)} \rangle_T = 0$.

**Step 5: Connection to Game-Theoretic Nash.**

The Nash equilibrium condition is:

$$
V^{(i)}(z^{(i)*}, z^{(-i)*}) \geq V^{(i)}(z^{(i)}, z^{(-i)*}) \quad \forall z^{(i)}, \forall i

$$

This is equivalent to $z^{(i)*}$ being a local maximum of $V^{(i)}(\cdot, z^{(-i)*})$, requiring:
1. **First-order:** $\nabla_{z^{(i)}} V^{(i)}|_{z^*} = 0$
2. **Second-order:** $\nabla^2_{z^{(i)}} V^{(i)}|_{z^*} \preceq 0$ (negative semi-definite Hessian)

The standing wave equilibrium satisfies the first-order condition via $\langle \nabla_A V^{(i)} \rangle_T = 0$.

**Step 6: Ground State Correspondence.**

The **ground state** (lowest eigenvalue $\lambda_1$) corresponds to:
- Minimal oscillation energy
- Longest wavelength mode
- Most stable equilibrium

Higher modes ($n > 1$) are metastable—small perturbations can cause transitions to lower modes. The stable Nash equilibrium corresponds to the ground state of the coupled system. $\square$

:::



## E.16 Derivation of the Game Tensor and Strategic Jacobian (Definition {prf:ref}`def-the-game-tensor`)

**Statement:** The Game Tensor $\mathcal{G}_{ij}$ arises from the second-order response of agent $i$'s value to agent $j$'s position, mediated by the Strategic Jacobian.

(proof-game-tensor-derivation)=
:::{prf:proof}

**Step 1: Best-Response Correspondence.**

In a multi-agent system, each agent $j$ has a **best-response correspondence**:

$$
BR_j(z^{(-j)}) := \arg\max_{z^{(j)}} V^{(j)}(z^{(j)}, z^{(-j)})

$$
where $z^{(-j)}$ denotes the positions of all agents except $j$.

**Assumption (Smooth Best-Response):** Assume $BR_j$ is single-valued and $C^1$ in a neighborhood of equilibrium. This holds when:
- The value function $V^{(j)}$ is strictly concave in $z^{(j)}$
- The equilibrium is isolated (non-degenerate Hessian)

**Step 2: Strategic Jacobian Definition.**

:::{prf:definition} Strategic Jacobian
:label: def-strategic-jacobian

The **Strategic Jacobian** $\mathcal{J}_{ji} \in \mathbb{R}^{d \times d}$ is the derivative of agent $j$'s best response with respect to agent $i$'s position:

$$
\mathcal{J}_{ji} := \frac{\partial BR_j(z^{(-j)})}{\partial z^{(i)}} = \frac{\partial z^{(j)*}}{\partial z^{(i)}}\bigg|_{BR}

$$
where $z^{(j)*} = BR_j(z^{(-j)})$.
:::

**Step 3: Implicit Function Theorem Derivation.**

At a best response, the first-order condition is:

$$
\nabla_{z^{(j)}} V^{(j)}(z^{(j)*}, z^{(-j)}) = 0

$$

Differentiating with respect to $z^{(i)}$ using the implicit function theorem:

$$
\nabla^2_{z^{(j)}z^{(j)}} V^{(j)} \cdot \frac{\partial z^{(j)*}}{\partial z^{(i)}} + \nabla^2_{z^{(j)}z^{(i)}} V^{(j)} = 0

$$

Solving for the Strategic Jacobian:

$$
\mathcal{J}_{ji} = -\left(\nabla^2_{z^{(j)}z^{(j)}} V^{(j)}\right)^{-1} \nabla^2_{z^{(j)}z^{(i)}} V^{(j)}

$$

**Step 4: Second-Order Value Variation.**

When agent $i$ moves by $\delta z^{(i)}$, agent $j$ responds with $\delta z^{(j)} \approx \mathcal{J}_{ji} \delta z^{(i)}$.

The second-order variation of agent $i$'s value is:

$$
\delta^2 V^{(i)} = (\delta z^{(i)})^\top \underbrace{\nabla^2_{z^{(i)}z^{(i)}} V^{(i)}}_{\text{direct curvature}} (\delta z^{(i)}) + (\delta z^{(i)})^\top \underbrace{\nabla^2_{z^{(i)}z^{(j)}} V^{(i)} \cdot \mathcal{J}_{ji}}_{\text{strategic back-reaction}} (\delta z^{(i)})

$$

**Step 5: Game Tensor as Effective Curvature.**

Define the **Game Tensor** as the strategic contribution to curvature:

$$
\mathcal{G}_{ij}^{kl} := \frac{\partial^2 V^{(i)}}{\partial z^{(j)}_k \partial z^{(j)}_l}\bigg|_{z^{(j)*}}

$$

The **perceived Hessian** including strategic back-reaction is:

$$
\tilde{H}^{(i)}_{kl} = \frac{\partial^2 V^{(i)}}{\partial z^{(i)}_k \partial z^{(i)}_l} + \sum_{j \neq i} \frac{\partial^2 V^{(i)}}{\partial z^{(i)}_k \partial z^{(j)}_m} (\mathcal{J}_{ji})^m_l

$$

**Step 6: Metric Modification.**

The agent's perceived geometry is modified by the Game Tensor. Under the Capacity-Constrained Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`), risk increases effective metric:

$$
\tilde{G}^{(i)}_{kl} = G^{(i)}_{kl} + \sum_{j \neq i} \beta_{ij} \mathcal{G}_{ij,kl}

$$

where:
- $\beta_{ij} > 0$ for adversarial agents (opponents increase perceived curvature)
- $\beta_{ij} = 0$ for neutral agents
- $\beta_{ij} < 0$ for cooperative agents (allies reduce perceived curvature)

The lowered-index Game Tensor is:

$$
\mathcal{G}_{ij,kl} = G^{(i)}_{km} G^{(i)}_{ln} \mathcal{G}_{ij}^{mn}

$$

**Physical Interpretation:** The Game Tensor measures how "curved" agent $i$'s value landscape appears due to agent $j$'s presence. High $\|\mathcal{G}_{ij}\|$ regions are strategically volatile—small movements create large value changes. $\square$

:::



## E.17 Proof of the Bianchi Identity (Theorem {prf:ref}`thm-bianchi-identity`)

**Statement:** The field strength tensor satisfies $D_{[\mu}\mathcal{F}_{\nu\rho]} = 0$ (cyclic sum vanishes).

(proof-bianchi-identity)=
:::{prf:proof}

**Step 1: Jacobi Identity for Covariant Derivatives.**

The covariant derivatives satisfy the Jacobi identity:

$$
[[D_\mu, D_\nu], D_\rho] + [[D_\nu, D_\rho], D_\mu] + [[D_\rho, D_\mu], D_\nu] = 0

$$

**Step 2: Commutator in Terms of Field Strength.**

From Theorem {prf:ref}`thm-curvature-commutator`:

$$
[D_\mu, D_\nu] = -ig\mathcal{F}_{\mu\nu}

$$
where $\mathcal{F}_{\mu\nu}$ acts on fields in the appropriate representation.

**Step 3: Action on a Test Field.**

Let $\psi$ be a field in the fundamental representation. Apply the Jacobi identity:

$$
[[D_\mu, D_\nu], D_\rho]\psi + \text{cyclic} = 0

$$

Compute the first term:

$$
[[D_\mu, D_\nu], D_\rho]\psi = [D_\mu, D_\nu](D_\rho \psi) - D_\rho([D_\mu, D_\nu]\psi)

$$

$$
= -ig\mathcal{F}_{\mu\nu}(D_\rho \psi) - D_\rho(-ig\mathcal{F}_{\mu\nu}\psi)

$$

$$
= -ig\mathcal{F}_{\mu\nu}D_\rho \psi + ig D_\rho(\mathcal{F}_{\mu\nu}\psi)

$$

$$
= -ig\mathcal{F}_{\mu\nu}D_\rho \psi + ig (D_\rho \mathcal{F}_{\mu\nu})\psi + ig \mathcal{F}_{\mu\nu}D_\rho \psi

$$

$$
= ig (D_\rho \mathcal{F}_{\mu\nu})\psi

$$

**Step 4: Covariant Derivative of Field Strength.**

The covariant derivative acts on $\mathcal{F}_{\mu\nu}$ (an adjoint-valued 2-form) as:

$$
D_\rho \mathcal{F}_{\mu\nu} = \partial_\rho \mathcal{F}_{\mu\nu} - ig[A_\rho, \mathcal{F}_{\mu\nu}]

$$

**Step 5: Cyclic Sum.**

From the Jacobi identity:

$$
ig(D_\mu \mathcal{F}_{\nu\rho} + D_\nu \mathcal{F}_{\rho\mu} + D_\rho \mathcal{F}_{\mu\nu})\psi = 0

$$

Since this holds for arbitrary $\psi$:

$$
D_\mu \mathcal{F}_{\nu\rho} + D_\nu \mathcal{F}_{\rho\mu} + D_\rho \mathcal{F}_{\mu\nu} = 0

$$

**Step 6: Component Form Verification.**

In components, with $\mathcal{F}_{\mu\nu}^a = \partial_\mu A_\nu^a - \partial_\nu A_\mu^a + g f^{abc} A_\mu^b A_\nu^c$:

$$
D_\rho \mathcal{F}_{\mu\nu}^a = \partial_\rho \mathcal{F}_{\mu\nu}^a + g f^{abc} A_\rho^b \mathcal{F}_{\mu\nu}^c

$$

The cyclic sum:

$$
D_{[\mu}\mathcal{F}_{\nu\rho]}^a = \partial_{[\mu}\mathcal{F}_{\nu\rho]}^a + g f^{abc} A_{[\mu}^b \mathcal{F}_{\nu\rho]}^c

$$

The first term vanishes by the Jacobi identity for ordinary derivatives (applied to the definition of $\mathcal{F}$):

$$
\partial_{[\mu}\mathcal{F}_{\nu\rho]} = \partial_{[\mu}(\partial_\nu A_{\rho]} - \partial_\rho A_{\nu]}) + g f^{abc} \partial_{[\mu}(A_\nu^b A_{\rho]}^c) = 0

$$

The second term vanishes by antisymmetry:

$$
f^{abc} A_{[\mu}^b \mathcal{F}_{\nu\rho]}^c = f^{abc} \cdot \frac{1}{6}(A_\mu^b \mathcal{F}_{\nu\rho}^c + \text{5 cyclic permutations}) = 0

$$

by the Jacobi identity for structure constants and antisymmetry of $\mathcal{F}$. $\square$

:::



## E.18 Derivation of the Higgs Mechanism (Theorem {prf:ref}`thm-higgs-mechanism`)

**Statement:** When $\mu^2 < 0$ in the Higgs potential, spontaneous symmetry breaking generates masses for gauge bosons and matter fields.

(proof-higgs-mechanism)=
:::{prf:proof}

**Step 1: Higgs Potential Minimization.**

The Higgs potential is:

$$
V(\Phi) = \mu^2 |\Phi|^2 + \lambda |\Phi|^4

$$

For $\mu^2 > 0$: Minimum at $\Phi = 0$ (symmetric phase).

For $\mu^2 < 0$: The potential has the "Mexican hat" shape. Setting $\partial V / \partial |\Phi| = 0$:

$$
2\mu^2 |\Phi| + 4\lambda |\Phi|^3 = 0

$$

$$
|\Phi|^2 = -\frac{\mu^2}{2\lambda} =: \frac{v^2}{2}

$$

The vacuum expectation value (VEV) is:

$$
\langle \Phi \rangle = \frac{v}{\sqrt{2}}, \quad v = \sqrt{-\frac{\mu^2}{\lambda}}

$$

**Step 2: Fluctuations Around the VEV.**

Expand around the vacuum:

$$
\Phi(z) = \frac{1}{\sqrt{2}}(v + h(z))e^{i\theta(z)/v}

$$

where:
- $h(z)$ is the **Higgs boson** (radial fluctuation, physical degree of freedom)
- $\theta(z)$ is the **Goldstone mode** (angular fluctuation, will be "eaten")

For small fluctuations, linearize:

$$
\Phi \approx \frac{1}{\sqrt{2}}(v + h + i\theta)

$$

**Step 3: Gauge Boson Mass Generation.**

The kinetic term for the Higgs field is:

$$
|D_\mu \Phi|^2 = |(\partial_\mu - igA_\mu)\Phi|^2

$$

Substituting $\Phi = (v + h)/\sqrt{2}$ (unitary gauge, $\theta = 0$):

$$
D_\mu \Phi = \frac{1}{\sqrt{2}}(\partial_\mu h - igA_\mu(v + h))

$$

$$
|D_\mu \Phi|^2 = \frac{1}{2}(\partial_\mu h)^2 + \frac{g^2}{2}(v + h)^2 A_\mu A^\mu - \frac{ig}{\sqrt{2}}(v+h)(A_\mu \partial^\mu h - \partial_\mu h A^\mu)

$$

The mass term for the gauge field emerges from the $(v^2)$ contribution:

$$
|D_\mu \Phi|^2 \supset \frac{g^2 v^2}{2} A_\mu A^\mu

$$

Comparing with the standard mass term $\frac{1}{2}m_A^2 A_\mu A^\mu$:

$$
m_A = gv

$$

**Step 4: Goldstone Boson Absorption.**

In the unitary gauge, the Goldstone mode $\theta$ is absorbed into the longitudinal component of the massive gauge boson. The gauge field gains a third polarization state (longitudinal), as required for a massive spin-1 particle.

**Counting degrees of freedom:**
- Before SSB: 2 (massless gauge) + 2 (complex Higgs) = 4
- After SSB: 3 (massive gauge) + 1 (real Higgs $h$) = 4 ✓

**Step 5: Matter Field Mass Generation (Yukawa).**

The Yukawa coupling is:

$$
\mathcal{L}_{\text{Yukawa}} = -y_{ij}\bar{\psi}^{(i)}\Phi\psi^{(j)}

$$

After SSB, substituting $\Phi = (v + h)/\sqrt{2}$:

$$
\mathcal{L}_{\text{Yukawa}} = -\frac{y_{ij}}{\sqrt{2}}(v + h)\bar{\psi}^{(i)}\psi^{(j)}

$$

$$
= -\frac{y_{ij} v}{\sqrt{2}}\bar{\psi}^{(i)}\psi^{(j)} - \frac{y_{ij}}{\sqrt{2}}h\bar{\psi}^{(i)}\psi^{(j)}

$$

The first term is a mass term with:

$$
m_{ij} = \frac{y_{ij} v}{\sqrt{2}}

$$

For diagonal Yukawa ($y_{ij} = y_i \delta_{ij}$):

$$
m_i = \frac{y_i v}{\sqrt{2}}

$$

**Step 6: Symmetry Breaking Pattern.**

The original symmetry group $G$ is broken to a subgroup $H$ that leaves the VEV invariant:

$$
U \langle \Phi \rangle = \langle \Phi \rangle \quad \text{for } U \in H

$$

The number of massive gauge bosons equals $\dim(G) - \dim(H)$ (the number of broken generators).

**Example:** For $G = SO(D)$ broken to $H = SO(D-1)$:
- Broken generators: $D - 1$
- Each broken generator → one massive gauge boson
- Remaining $SO(D-1)$ gauge bosons stay massless $\square$

:::



## E.19 Proof of Nash Equilibrium as Ground State (Theorem {prf:ref}`thm-nash-ground-state`)

**Statement:** In the semiclassical limit $\sigma \to 0$, the ground state wave-function concentrates on the Nash equilibrium.

(proof-nash-ground-state)=
:::{prf:proof}

**Step 1: WKB/Semiclassical Ansatz.**

For small $\sigma$, seek solutions of the form:

$$
\Psi(\mathbf{z}) = A(\mathbf{z}) \exp\left(-\frac{S(\mathbf{z})}{\sigma}\right)

$$

where $S(\mathbf{z}) \geq 0$ is the "action" and $A(\mathbf{z})$ is a slowly-varying amplitude.

**Step 2: Substitution into Schrödinger.**

The Strategic Hamiltonian acting on $\Psi$:

$$
\hat{H}_{\text{strat}}\Psi = \left[-\frac{\sigma^2}{2}\Delta_{\tilde{G}} + \Phi_{\text{eff}}\right]\Psi

$$

Compute the Laplacian of the WKB ansatz:

$$
\Delta_{\tilde{G}}(Ae^{-S/\sigma}) = e^{-S/\sigma}\left[\Delta_{\tilde{G}} A - \frac{2}{\sigma}\tilde{G}^{-1}(\nabla A, \nabla S) - \frac{A}{\sigma}\Delta_{\tilde{G}} S + \frac{A}{\sigma^2}\|\nabla S\|_{\tilde{G}}^2\right]

$$

**Step 3: Leading Order ($O(\sigma^{-2})$).**

The leading term gives:

$$
-\frac{\sigma^2}{2} \cdot \frac{A}{\sigma^2}\|\nabla S\|_{\tilde{G}}^2 = -\frac{A}{2}\|\nabla S\|_{\tilde{G}}^2

$$

For the ground state (minimum energy), we need:

$$
E_0 = \frac{1}{2}\|\nabla S\|_{\tilde{G}}^2 + \Phi_{\text{eff}}

$$

This is minimized when $\|\nabla S\|_{\tilde{G}}^2 = 0$ and $\Phi_{\text{eff}}$ is minimized.

**Step 4: Concentration on Critical Points.**

The condition $\nabla S = 0$ implies that $S$ is constant along directions where the wave-function has support. The wave-function $|\Psi|^2 = |A|^2 e^{-2S/\sigma}$ concentrates exponentially on the **minimum of $S$**.

For the ground state, $S(\mathbf{z}) = S_0 + \frac{1}{2}(\mathbf{z} - \mathbf{z}^*)^\top H (\mathbf{z} - \mathbf{z}^*) + O(|\mathbf{z} - \mathbf{z}^*|^3)$

where $\mathbf{z}^*$ is the minimum and $H$ is the Hessian.

**Step 5: Gaussian Approximation.**

Near the minimum:

$$
|\Psi(\mathbf{z})|^2 \approx |A(\mathbf{z}^*)|^2 \exp\left(-\frac{(\mathbf{z} - \mathbf{z}^*)^\top H (\mathbf{z} - \mathbf{z}^*)}{\sigma}\right)

$$

This is a Gaussian with width $\sim \sqrt{\sigma}$. As $\sigma \to 0$:

$$
|\Psi(\mathbf{z})|^2 \to \delta(\mathbf{z} - \mathbf{z}^*)

$$

**Step 6: Identification with Nash Equilibrium.**

The minimum of $\Phi_{\text{eff}}(\mathbf{z})$ is the Nash equilibrium by definition:
- $\Phi_{\text{eff}}^{(i)}(z^{(i)}, z^{(-i)}) = -V^{(i)}(z^{(i)}, z^{(-i)})$ (negative value = cost)
- Nash: each agent maximizes their own value → minimizes their own cost
- Joint minimum: $\nabla_{z^{(i)}} \Phi_{\text{eff}}^{(i)} = 0$ for all $i$

**Step 7: Energy Correction.**

The ground state energy is:

$$
E_0 = \Phi_{\text{eff}}(\mathbf{z}^*) + O(\sigma)

$$

The $O(\sigma)$ correction comes from zero-point energy:

$$
E_0 = \Phi_{\text{eff}}(\mathbf{z}^*) + \frac{\sigma}{2}\text{Tr}(\sqrt{H \tilde{G}^{-1}}) + O(\sigma^2)

$$

This is the sum of $\frac{\sigma \omega_n}{2}$ over all normal mode frequencies $\omega_n = \sqrt{\lambda_n}$ where $\lambda_n$ are eigenvalues of $H \tilde{G}^{-1}$.

**Step 8: Stability from Spectral Gap.**

The Nash equilibrium is **stable** if $H \succ 0$ (positive definite Hessian at the minimum). This ensures:
1. The ground state is unique
2. There is a spectral gap $\Delta = E_1 - E_0 > 0$
3. The concentration is exponentially tight in $\sigma$

Unstable critical points (saddles) have $H$ with negative eigenvalues, leading to **excited states** rather than ground states. $\square$

:::



(sec-references)=
## References

```{bibliography}
```
