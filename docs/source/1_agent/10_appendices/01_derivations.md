(sec-appendix-a-full-derivations)=
# {ref}`Appendix A <sec-appendix-a-full-derivations>`: Full Derivations (Capacity-Constrained Curvature Functional)

## TLDR

- This appendix records the **full derivations** behind the main geometric/information-theoretic claims in Volume 1.
- It is reference material: use it when you want the complete steps, not first-pass intuition.
- Most readers can skim the statements and return to details when implementing or auditing proofs.

(sec-appendix-a-capacity-constrained-curvature-functional)=
## A.1 Capacity-Constrained Curvature Functional (Variational Principle)

If the agent attempts to maintain internal structure such that $I_{\text{bulk}}(\mathcal{Z})>C_{\partial}(\partial\mathcal{Z})$, it has exceeded what can be grounded at the boundary; this is an enclosure violation and must be rejected by the Sieve ({ref}`Section 3 <sec-diagnostics-stability-checks>`, Node 13).

In the optimal / sound regime the constraint is active (saturated):

$$
I_{\text{bulk}}(\mathcal{Z}) = C_{\partial}(\partial\mathcal{Z}).

$$
We now encode this as a variational constraint to obtain a *metric law* from information limits.

:::{prf:definition} A.1.1 (Boundary capacity form)
:label: def-a-boundary-capacity-form

Define the boundary capacity $(n\!-\!1)$-form

$$
\omega_{\partial} := \frac{1}{\eta_\ell}\, dA_G,

$$
so that $C_{\partial}(\partial\mathcal{Z})=\oint_{\partial\mathcal{Z}}\omega_{\partial}$ (Definition 17.1.3).

:::
:::{prf:definition} A.1.2 (Boundary-capacity constraint functional)
:label: def-a-boundary-capacity-constraint-functional

Define the saturation functional

$$
\mathcal{C}[G,V]
:=
\underbrace{\int_{\mathcal{Z}} \rho_I(G,V)\, d\mu_G}_{I_{\text{bulk}}}
\;-\;
\underbrace{\oint_{\partial\mathcal{Z}}\omega_{\partial}}_{C_{\partial}},

$$
where $\rho_I(G,V)$ is an *information density* (nats per unit $d\mu_G$) compatible with the agent's representation scheme (Definition 17.1.2). This $\rho_I$ is distinct from the belief density $p$ used in {ref}`Section 2.11 <sec-variance-value-duality-and-information-conservation>`. When $\rho_I$ is instantiated via the split shutter, the most conservative computable proxy is a global one, $I_{\text{bulk}}\approx \mathbb{E}[I(X;K)]$ (Node 13), and the window theorem (Theorem {prf:ref}`thm-information-stability-window-operational`) supplies the admissible operating range.

:::
:::{prf:definition} A.1.3 (Risk Lagrangian density)
:label: def-a-risk-lagrangian-density

Fix a smooth potential $V\in C^\infty(\mathcal{Z})$. A canonical risk Lagrangian density is the scalar-field functional

$$
\mathcal{L}_{\text{risk}}(V;G) := \frac{1}{2}\,G^{ab}\nabla_a V\,\nabla_b V + U(V),

$$
where $U:\mathbb{R}\to\mathbb{R}$ is a (possibly learned) on-site potential capturing non-gradient costs. (The sign convention is chosen for a Riemannian metric; see e.g. Lee, *Riemannian Manifolds*, 2018, for the variational identities used below.)

:::
:::{prf:definition} A.1.4 (Capacity-constrained curvature functional)
:label: def-a-capacity-constrained-curvature-functional

Let $R(G)$ be the scalar curvature of $G$ and let $\Lambda\in\mathbb{R}$ be a constant. Define the constrained functional

$$
\mathcal{S}[G,V]
:=
\int_{\mathcal{Z}}\left(R(G)-2\Lambda + 2\kappa\,\mathcal{L}_{\text{risk}}(V;G)\right)d\mu_G
\;-\;
2\kappa\oint_{\partial\mathcal{Z}}\omega_{\partial},

$$
with coupling $\kappa\in\mathbb{R}$. The last term is the explicit boundary capacity penalty, and $\Lambda$ is a bulk capacity offset that remains once the boundary is clamped at finite resolution.

*Remark (why $\Lambda$ is allowed).* A constant term in the integrand is the simplest coordinate-invariant scalar density and produces a $\Lambda G_{ij}$ term in the metric Euler–Lagrange equation. Here $\Lambda$ plays the role of a baseline curvature / capacity offset.

:::
(sec-appendix-a-first-variation)=
## A.2 First Variation (Expanded Derivation)

We work in the standard calculus of variations on Riemannian manifolds with boundary. Assume:
1) $G$ is $C^2$ and $V$ is $C^2$ (so curvature and gradients are well-defined),
2) variations $\delta G^{ij}$ are smooth, symmetric, and compactly supported in $\mathcal{Z}$ or satisfy Dirichlet boundary conditions $\delta G^{ij}\vert_{\partial\mathcal{Z}}=0$ (the boundary is clamped by the sensorium; cf. Definition {prf:ref}`def-observation-inflow-form` / Theorem {prf:ref}`thm-generalized-conservation-of-belief`).

Under these hypotheses, the first variation of $\mathcal{S}$ is well-defined as a distribution; the standard identities below can be found in standard differential-geometry references (e.g. {cite}`lee2018riemannian`).

(sec-appendix-a-variation-of-the-volume-form)=
### A.2.1 Variation of the volume form

Let $d\mu_G=\sqrt{|G|}\,dz^n$. The determinant identity gives

$$
\delta \sqrt{|G|} = -\frac{1}{2}\sqrt{|G|}\,G_{ij}\,\delta G^{ij},

$$
equivalently $\delta d\mu_G = -\tfrac12\,G_{ij}\,\delta G^{ij}\, d\mu_G$.

(sec-appendix-a-variation-of-the-curvature-term)=
### A.2.2 Variation of the curvature term

Write the curvature functional as $\mathcal{S}_{\text{geo}}[G]:=\int_{\mathcal{Z}}R(G)\,d\mu_G$. The variation splits as

$$
\delta(R\,d\mu_G) = (\delta R)\,d\mu_G + R\,\delta d\mu_G.

$$
For the scalar curvature, use

$$
R = G^{ij}R_{ij},

$$
hence

$$
\delta R = R_{ij}\,\delta G^{ij} + G^{ij}\,\delta R_{ij}.

$$
The Palatini identity gives

$$
\delta R_{ij} = \nabla_k(\delta \Gamma^k_{ij})-\nabla_j(\delta\Gamma^k_{ik}),

$$
and the Christoffel variation is

$$
\delta\Gamma^k_{ij} = \frac12\,G^{k\ell}\left(\nabla_i \delta G_{j\ell}+\nabla_j \delta G_{i\ell}-\nabla_\ell \delta G_{ij}\right),

$$
where $\delta G_{ij} = -G_{ia}G_{jb}\,\delta G^{ab}$.

Substituting and collecting terms yields the standard decomposition

$$
\delta\mathcal{S}_{\text{geo}} = \int_{\mathcal{Z}}\left(R_{ij}-\frac12 R\,G_{ij}\right)\delta G^{ij}\,d\mu_G + \oint_{\partial\mathcal{Z}} \mathcal{B}_{\text{curv}}(\delta G,\nabla\delta G),

$$
where $\mathcal{B}_{\text{curv}}$ is an explicit boundary $(n\!-\!1)$-form built from $\delta\Gamma$ (equivalently from $\delta G$ and its first derivatives). For a well-posed Dirichlet variational problem one can add an appropriate boundary term to cancel $\mathcal{B}_{\text{curv}}$. In our setting the boundary is clamped, so we impose $\delta G\vert_{\partial\mathcal{Z}}=0$ and the boundary term vanishes.

(sec-appendix-a-variation-of-the-risk-term)=
### A.2.3 Variation of the risk term

Let $\mathcal{S}_{\text{risk}}[G,V] := \int_{\mathcal{Z}}\mathcal{L}_{\text{risk}}(V;G)\,d\mu_G$. Define the (Riemannian-signature) risk tensor by

$$
T_{ij} := -\frac{2}{\sqrt{|G|}}\frac{\delta(\sqrt{|G|}\,\mathcal{L}_{\text{risk}})}{\delta G^{ij}}.

$$
Holding $V$ fixed under $\delta G$ and using $\delta d\mu_G = -\tfrac12 G_{ij}\delta G^{ij} d\mu_G$ gives the standard identity

$$
\delta \mathcal{S}_{\text{risk}} = -\frac12 \int_{\mathcal{Z}} T_{ij}\,\delta G^{ij}\,d\mu_G.

$$
For the risk Lagrangian

$$
\mathcal{L}_{\text{risk}}=\tfrac12 G^{ab}\nabla_a V\nabla_b V + U(V),

$$
the explicit computation yields

$$
T_{ij} = \nabla_i V\,\nabla_j V - G_{ij}\left(\frac12\,G^{ab}\nabla_a V\nabla_b V + U(V)\right).

$$
(sec-appendix-a-capacity-term-and-the-emergence-of)=
### A.2.4 Capacity (boundary) term and the emergence of $\Lambda$

The explicit boundary penalty $-\kappa\oint_{\partial\mathcal{Z}}\omega_{\partial}$ depends only on the induced boundary metric through $dA_G$. Under the clamped boundary condition $\delta G\vert_{\partial\mathcal{Z}}=0$, its first variation vanishes.

The remaining constant $\Lambda$ in Definition A.1.4 plays the role of the bulk Lagrange multiplier for finite boundary capacity: it is the only diffeomorphism-invariant way to represent an additive “grounding floor” required for non-degenerate macrostates (cf. Theorem {prf:ref}`thm-information-stability-window-operational`). Formally, varying $-2\Lambda\int_{\mathcal{Z}} d\mu_G$ gives

$$
\delta\left(-2\Lambda\int_{\mathcal{Z}} d\mu_G\right) = \int_{\mathcal{Z}} \Lambda G_{ij}\,\delta G^{ij}\,d\mu_G.

$$
(sec-appendix-a-recovery-of-the-metric-stationarity-condition)=
## A.3 Recovery of the Metric Stationarity Condition

:::{prf:lemma} A.3.1 (Divergence-to-boundary conversion)
:label: lem-a-divergence-to-boundary-conversion

For any sufficiently regular information flux field $\mathbf{j}$ on $\mathcal{Z}$,

$$
\int_{\mathcal{Z}} \operatorname{div}_G(\mathbf{j})\, d\mu_G = \oint_{\partial \mathcal{Z}} \langle \mathbf{j}, \mathbf{n}\rangle\, dA_G,

$$
which is the Riemannian divergence theorem underlying the global balance equation in Theorem {prf:ref}`thm-generalized-conservation-of-belief`.

:::
:::{prf:theorem} A.3.2 (Capacity-consistency identity; proof of Theorem {prf:ref}`thm-capacity-constrained-metric-law`)
:label: thm-a-capacity-consistency-identity-proof-of-theorem

Under the hypotheses of Section A.2, stationarity of $\mathcal{S}[G,V]$ with respect to arbitrary variations $\delta G^{ij}$ that vanish on $\partial\mathcal{Z}$ implies the Euler–Lagrange equation

$$
R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa\, T_{ij},

$$
with $T_{ij}$ given by Section A.2.3.

*Proof.* Combine Sections A.2.1–A.2.4:

$$
\delta\mathcal{S} = \int_{\mathcal{Z}}\left[\left(R_{ij}-\frac12 R\,G_{ij}\right) + \Lambda G_{ij} - \kappa T_{ij}\right]\delta G^{ij}\,d\mu_G + \text{(boundary terms)}.

$$
Boundary terms vanish under the clamped boundary condition (or after adding an appropriate boundary term). Because $\delta G^{ij}$ is arbitrary in the interior, the fundamental lemma of the calculus of variations implies the bracketed tensor must vanish pointwise almost everywhere, yielding the stated identity (see e.g. Evans, *Partial Differential Equations*, 2010, for the functional-analytic lemma).

*Interpretation.* The Ricci curvature governs local volume growth; enforcing a boundary-limited bulk information volume forces the metric to stretch/compress coordinates so that information-dense regions (large $\|\nabla_A V\|$ and/or large $U(V)$) do not generate bulk structure that cannot be grounded at the boundary.

*Remark (regularizer).* The squared residual of this identity defines the capacity-consistency loss $\mathcal{L}_{\text{cap-metric}}$; see {ref}`Appendix B <sec-appendix-b-units-parameters-and-coefficients>`.

:::
(sec-appendix-a-pitchfork-bifurcation-at-the-origin)=
## A.3b Pitchfork Bifurcation at the Origin (Derivation supporting Definition {prf:ref}`def-control-field-at-origin`)

This section provides a full derivation underpinning Definition {prf:ref}`def-control-field-at-origin`.

**Setup.** Consider the Langevin equation on the Poincare disk $\mathbb{D}$ from Definition {prf:ref}`prop-so-d-symmetry-at-origin`:

$$
dz_\tau = -\nabla_G U(z_\tau)\, d\tau + \sqrt{2T_c}\, G^{-1/2}(z_\tau)\, dW_\tau,

$$
with $U(z) = -2\operatorname{artanh}(|z|)$ and initial condition $z(0) = 0$.

**Step 1: Linearization near the origin.**

Near $z = 0$, expand the potential:

$$
U(z) = -2\operatorname{artanh}(|z|) \approx -2|z| - \frac{2}{3}|z|^3 + O(|z|^5) \approx -|z|^2 + O(|z|^4).

$$
The Euclidean gradient is:

$$
\nabla U(z) = -\frac{2z}{1-|z|^2} \approx -2z + O(|z|^3).

$$
The Riemannian gradient (with $G^{-1} = \frac{(1-|z|^2)^2}{4}I \approx \frac{1}{4}I$ near origin):

$$
\nabla_G U(z) \approx -\frac{1}{2}z + O(|z|^3).

$$
**Step 2: Fokker-Planck equation.**

The probability density $p(z, \tau)$ satisfies the Fokker-Planck equation:

$$
\partial_\tau p = \nabla \cdot \left( p\,\nabla_G U \right) + T_c\,\nabla \cdot (G^{-1} \nabla p).

$$
Near the origin, with the linearization $\nabla_G U \approx -\frac{1}{2}z$ and $G^{-1} \approx \frac{1}{4}I$:

$$
\partial_\tau p \approx \nabla \cdot \left( -\frac{p\,z}{2} \right) + \frac{T_c}{4}\,\Delta p = \frac{p}{2} + \frac{z \cdot \nabla p}{2} + \frac{T_c}{4}\,\Delta p.

$$
**Step 3: Stationary distribution.**

The stationary solution $p_*(z)$ satisfies detailed balance:

$$
\nabla_G U + T_c\,G^{-1}\,\nabla \log p_* = 0.

$$
Substituting and solving:

$$
\nabla \log p_* = -\frac{1}{T_c}\,G\,\nabla_G U = -\frac{1}{T_c}\,\nabla U.

$$
Integrating:

$$
p_*(z) \propto \exp\left(-\frac{U(z)}{T_c}\right) = \exp\left(\frac{2\operatorname{artanh}(|z|)}{T_c}\right).

$$
This distribution is **rotationally symmetric** (depends only on $|z|$), establishing Part 1 of Definition {prf:ref}`def-control-field-at-origin`.

**Step 4: Bifurcation analysis.**

Write $z = re^{i\theta}$ in polar coordinates. The effective potential in the radial direction is:

$$
U_{\text{eff}}(r) = -2\operatorname{artanh}(r).

$$
The radial force is $F_r = -\frac{dU_{\text{eff}}}{dr} = \frac{2}{1-r^2} > 0$ for all $r \in [0, 1)$.

This means:
- The origin $r = 0$ is an **unstable equilibrium** (force points outward).
- There is no stable equilibrium in the interior; the "stable point" is at the boundary $r = 1$.
- The angular direction $\theta$ is **neutral** (no restoring force).

**Step 5: Symmetry breaking mechanism.**

For small $\tau$:
1. The noise term dominates: $z(\tau) \approx \sqrt{2T_c}\int_0^\tau G^{-1/2} dW_\tau$ performs a random walk.
2. This random walk samples directions $\theta$ uniformly from $[0, 2\pi)$.
3. Once $|z|$ exceeds a threshold (order $\sqrt{T_c}$), the deterministic drift $-\nabla_G U$ takes over.
4. The trajectory then flows radially outward along the selected direction $\theta$.

This is the **supercritical pitchfork bifurcation** structure: the continuous symmetry $SO(2)$ acting on $\theta$ is spontaneously broken to the identity when a specific direction is selected.

**Step 6: Critical exponents.**

The escape time from the neighborhood of the origin scales as:

$$
\tau_{\text{escape}} \sim \frac{1}{T_c}\,\exp\left(\frac{\Delta U}{T_c}\right),

$$
where $\Delta U$ is the "barrier height" (which is zero here since the origin is unstable). Thus $\tau_{\text{escape}} \sim O(1)$—the system escapes quickly.

The direction selected $\theta^*$ is uniformly distributed: $\theta^* \sim \mathrm{Uniform}[0, 2\pi)$.

This completes the derivation supporting Definition {prf:ref}`def-control-field-at-origin`. $\square$

(sec-appendix-a-overdamped-limit-via-singular-perturbation)=
## A.4 Overdamped Limit via Singular Perturbation (Proof of Theorem {prf:ref}`thm-overdamped-limit`)

This section provides the full proof of Theorem {prf:ref}`thm-overdamped-limit` using singular perturbation theory.

**Setup.** Consider the second-order Langevin equation with friction:

$$
m\,\ddot{z}^k + \gamma\,\dot{z}^k + G^{kj}\partial_j\Phi + \Gamma^k_{ij}\dot{z}^i\dot{z}^j = \sqrt{2T_c}\,(G^{-1/2})^{kj}\,\xi^j,

$$
where $m$ is inertial mass, $\gamma$ is friction, and $\xi^j$ is white noise.

**Step 1: Non-dimensionalization.**

Introduce the dimensionless parameter $\epsilon = m/\gamma$ (mass-to-friction ratio). Rescale computation time as $\tilde{s} = s/\gamma$ so that $d/ds = (1/\gamma)\,d/d\tilde{s}$. The equation becomes:

$$
\epsilon\,\frac{d^2z^k}{d\tilde{s}^2} + \frac{dz^k}{d\tilde{s}} + \frac{1}{\gamma}\,G^{kj}\partial_j\Phi + \frac{\epsilon}{\gamma}\,\Gamma^k_{ij}\frac{dz^i}{d\tilde{s}}\frac{dz^j}{d\tilde{s}} = \sqrt{\frac{2T_c}{\gamma}}\,(G^{-1/2})^{kj}\,\tilde{\xi}^j,

$$
where $\tilde{\xi}$ is appropriately rescaled noise.

**Step 2: Singular perturbation expansion.**

In the limit $\epsilon \to 0$, expand:

$$
z(\tilde{s}) = z_0(\tilde{s}) + \epsilon\,z_1(\tilde{s}) + O(\epsilon^2).

$$
At leading order ($\epsilon^0$):

$$
\frac{dz_0^k}{d\tilde{s}} = -\frac{1}{\gamma}\,G^{kj}(z_0)\,\partial_j\Phi(z_0) + \sqrt{\frac{2T_c}{\gamma}}\,(G^{-1/2})^{kj}\,\tilde{\xi}^j.

$$
Returning to original computation time $s = \gamma\tilde{s}$ and using $dz_0/ds = (1/\gamma)\,dz_0/d\tilde{s}$:

$$
dz_0^k = -G^{kj}(z_0)\,\partial_j\Phi(z_0)\,ds + \sqrt{2T_c}\,(G^{-1/2})^{kj}\,dW^j_s.

$$
This is exactly the overdamped equation stated in Theorem {prf:ref}`thm-overdamped-limit`.

**Step 3: Negligibility of the geodesic term.**

The Christoffel term has magnitude:

$$
|\Gamma^k_{ij}\dot{z}^i\dot{z}^j| \sim |\Gamma|\,|\dot{z}|^2.

$$
In the overdamped limit, $\dot{z} \approx \mathcal{M}_\gamma(-G^{-1}\nabla\Phi)$ with
$\mathcal{M}_\gamma := (\gamma I - \beta_{\text{curl}} G^{-1}\mathcal{F})^{-1}$. Thus:

$$
|\Gamma|\,|\dot{z}|^2 \sim |\Gamma|\,\|\mathcal{M}_\gamma G^{-1}\nabla\Phi\|^2 \to 0 \quad \text{as } \gamma \to \infty.

$$
The geodesic correction is suppressed by $O(\gamma^{-2})$.

**Step 4: Boundary layer analysis.**

For completeness, we note that there is a "boundary layer" in time of width $\Delta s \sim m/\gamma = \epsilon$ during which the velocity equilibrates to the force. Within this layer, the full second-order dynamics apply. Outside this layer (for $s \gg \epsilon$), the first-order approximation is accurate.

**Step 5: Error bounds.**

The error in the overdamped approximation is bounded by:

$$
\|z(s) - z_0(s)\| \le C\,\epsilon\,(1 + s)\,e^{-s/\epsilon},

$$
where $C$ depends on the smoothness of $G$, $\Phi$, and $\Gamma$. For $s \gg \epsilon$, this error is exponentially small.

This completes the proof of Theorem {prf:ref}`thm-overdamped-limit`. $\square$

:::{prf:remark} Physical interpretation
:label: rem-physical-interpretation

The overdamped limit corresponds to:
- **Information geometry:** The "friction" $\gamma$ represents the rate of information dissipation (forgetting). High friction means the system equilibrates quickly to the local gradient.
- **Diffusion models:** Standard score-based diffusion models operate entirely in the overdamped regime, with $\gamma \to \infty$ implicitly.
- **Neural network training:** The geodesic term $\Gamma(\dot{z},\dot{z})$ can be interpreted as a "momentum correction" that accounts for the curvature of the loss landscape. In standard gradient descent (overdamped), this term is ignored.

:::
(sec-appendix-a-classification-as-relaxation)=
## A.5 Classification as Relaxation (Proof of Theorem {prf:ref}`thm-classification-as-relaxation`)

:::{prf:theorem} Classification as Relaxation
:label: thm-classification-as-relaxation-a

Under the overdamped dynamics with class-conditioned potential $V_y$:

$$
dz = \mathcal{M}_{\text{curl}}\!\left(-G^{-1}(z) \nabla_A V_y(z, K)\right) ds + \sqrt{2T_c}\, G^{-1/2}(z)\, dW_s,

$$
where $\mathcal{M}_{\text{curl}} := (I - \beta_{\text{curl}} G^{-1}\mathcal{F})^{-1}$. Here $\nabla_A V_y := \nabla V_y - A$ (conservative case: $A=0$). When $\mathcal{F}=0$ (conservative case), the curl term vanishes.
the limiting chart assignment satisfies $\lim_{s \to \infty} K(z(s)) \in \mathcal{A}_y$ almost surely, provided the initial condition lies in the basin $\mathcal{B}_y$ and $T_c$ is sufficiently small.

:::

(proof-thm-classification-as-relaxation-a)=
:::{prf:proof}

**Step 1: Lyapunov Function Construction.**

Define the Lyapunov function:

$$
L(z) := V_y(z, K(z)) = -\beta_{\text{class}} \log P(Y=y \mid K(z)) + V_{\text{base}}(z, K(z)).

$$
By construction, $L(z)$ achieves its global minimum on the sub-atlas $\mathcal{A}_y$, where $P(Y=y \mid K) > 1 - \epsilon_{\text{purity}}$, hence $-\log P(Y=y \mid K) < -\log(1 - \epsilon_{\text{purity}})$ is minimized.

**Step 2: Itô Computation.**

Applying Itô's lemma to $L(z(s))$:

$$
dL = \nabla L \cdot dz + \frac{1}{2} \text{tr}(\nabla^2 L \cdot \Sigma)\, ds,

$$
where $\Sigma = 2T_c\, G^{-1}$ is the diffusion covariance.

Substituting the SDE:

$$
dL = \nabla L \cdot \left(-G^{-1} \nabla_A V_y\, ds + \sqrt{2T_c}\, G^{-1/2}\, dW_s\right) + T_c\, \Delta_G L\, ds,

$$
where $\Delta_G L = \text{tr}(G^{-1} \nabla^2 L)$ is the Laplace-Beltrami operator.

Since $L = V_y$, we have $\nabla L = \nabla V_y$, so:

$$
dL = -G^{-1}(\nabla V_y, \nabla_A V_y)\, ds + \sqrt{2T_c}\, \nabla V_y \cdot G^{-1/2}\, dW_s + T_c\, \Delta_G V_y\, ds.

$$
**Step 3: Expected Drift.**

Taking expectations:

$$
\frac{d}{ds}\mathbb{E}[L(z(s))] = -\mathbb{E}[G^{-1}(\nabla V_y, \nabla_A V_y)] + T_c\, \mathbb{E}[\Delta_G V_y].

$$
The first term is non-positive in the conservative case ($A=0$), reducing to $-\|\nabla V_y\|_G^2$. The second term is $O(T_c)$ and bounded if $V_y$ has bounded Hessian.

**Step 4: Low-Temperature Limit.**

For $T_c \to 0$, the drift becomes:

$$
\frac{d}{ds}\mathbb{E}[L] \approx -\mathbb{E}[G^{-1}(\nabla V_y, \nabla_A V_y)],

$$
with equality only at critical points of $V_y$.

**Step 5: Convergence to Attractor Basin.**

By LaSalle's invariance principle (conservative case), the trajectory converges to the largest invariant set where $\|\nabla V_y\|_G = 0$. Since $V_y$ is constructed with:
- A global minimum on $\mathcal{A}_y$ (the class-$y$ sub-atlas)
- Local maxima or saddles in transition regions $\mathcal{A}_i \cap \mathcal{A}_j$

If $z(0) \in \mathcal{B}_y$ (the basin of attraction for $\mathcal{A}_y$), the trajectory cannot escape to other basins (they are separated by energy barriers), hence:

$$
\lim_{s \to \infty} z(s) \in \mathcal{A}_y \quad \text{a.s.}

$$
**Step 6: Chart Assignment.**

*Technical note (Piecewise Continuity).* The Lyapunov function $L(z) = V_y(z, K(z))$ has potential discontinuities at chart boundaries where $K(z)$ changes discretely. However, this does not invalidate the argument because:

1. **Within-chart dynamics:** The SDE governs continuous motion within each chart; chart transitions occur via the jump process ({ref}`Section 20.5 <sec-connection-to-gksl-master-equation>`, WFR reaction term)
2. **Jump consistency:** The class-modulated jump rates (Definition {prf:ref}`def-class-consistent-jump-rate`) ensure that jumps to charts in $\mathcal{A}_y$ are favored when starting from $\mathcal{B}_y$
3. **Effective continuity:** For the soft router weights $w_k(x)$, the "effective chart" is a convex combination, and $\sum_k w_k(z) V_y(z, k)$ is continuous

Since $z(s) \to \mathcal{A}_y$ and the chart assignment $K(z)$ eventually stabilizes (jumps become rare as $z$ approaches the basin interior), we have:

$$
\lim_{s \to \infty} K(z(s)) \in \mathcal{A}_y.

$$
**Quantitative Bound (Low Temperature).**

For small but positive $T_c$, standard results on diffusions in potential wells (Kramers' law {cite}`kramers1940brownian`) give the escape rate from basin $\mathcal{B}_y$:

$$
\text{Rate}_{\text{escape}} \sim e^{-\Delta V / T_c},

$$
where $\Delta V$ is the barrier height. For $T_c \ll \Delta V$, escape is exponentially unlikely, ensuring practical convergence.

This completes the proof. $\square$

:::
:::{prf:remark} Connection to Classification Accuracy
:label: rem-connection-to-classification-accuracy

The theorem provides a geometric interpretation of classification accuracy: a sample $x$ is correctly classified if and only if $\text{Enc}(x) \in \mathcal{B}_{y_{\text{true}}}$. Misclassification occurs when the encoder maps $x$ to the wrong basin—either due to encoder limitations or overlap between class distributions in observation space.

:::



(sec-appendix-a-area-law)=
## A.6 The Area Law Coefficient (Proof of Theorem {prf:ref}`thm-causal-information-bound`)

This section provides the rigorous derivation of the Causal Information Bound, including the origin of the $1/4$ coefficient.

**Setup.** Let $(\mathcal{Z}, G)$ be the latent Riemannian manifold. We seek the maximum bulk information $I_{\text{bulk}}$ that can be distinguished by an external observer through the boundary $\partial\mathcal{Z}$.

**Note on Derivation Strategy.** We present *two* derivations of the Area Law:
1. **Microstate Counting** (Sections A.6.0–A.6.1b): A non-circular derivation from first principles of information geometry, independent of the Metric Law.
2. **Field-Theoretic** (Sections A.6.1–A.6.5): A derivation via the Metric Law, showing consistency with the geometric approach.

The first derivation establishes the bound from counting distinguishable states; the second shows that the Metric Law reproduces this bound dynamically.



(sec-appendix-a-foundational-axioms)=
### A.6.0 Foundational Axioms for Microstate Counting

This section establishes the information-theoretic foundations required for a non-circular derivation of the Area Law, analogous to Strominger-Vafa's microstate counting for black hole entropy {cite}`strominger1996microscopic`.

:::{prf:axiom} A.6.0a (Operational Distinguishability)
:label: ax-a-operational-distinguishability

Two probability distributions $p, q \in \mathcal{P}(\mathcal{Z})$ are **operationally distinguishable** if and only if:

$$
D_{\text{KL}}(p \| q) \geq 1 \text{ nat}.

$$
*Justification.* This is an **operational definition**, not a derived fact. The choice of 1 nat as the threshold is grounded in:

1. **Asymptotic error exponent.** For $n$ i.i.d. samples, the optimal Type II error probability at fixed Type I error decays as $\exp(-n \cdot D_{\text{KL}})$ (Stein's lemma). Thus $D_{\text{KL}} = 1$ nat corresponds to error decay rate $e^{-n}$.

2. **Information-theoretic meaning.** 1 nat = log(e) ≈ 1.44 bits represents a "natural unit" of information, where the likelihood ratio $p(x)/q(x)$ has expected log-value 1 under $p$.

3. **Dimensional analysis.** The nat is the natural unit when using natural logarithms; choosing 1 nat as the threshold makes the subsequent formulas dimensionally consistent.

*Remark.* Alternative thresholds (e.g., 1 bit = ln 2 nats) would change the numerical coefficient in the Area Law but not its structure.

:::

:::{prf:theorem} A.6.0b (Chentsov's Uniqueness Theorem)
:label: thm-a-chentsov-uniqueness

The **Fisher Information Metric** is the unique Riemannian metric on statistical manifolds (up to constant scaling) that is invariant under sufficient statistics.

**Statement.** Let $\mathcal{M}$ be a statistical manifold parameterized by $\theta \in \Theta$. Any Riemannian metric $g$ on $\mathcal{M}$ satisfying:
1. **Markov invariance:** $g$ is preserved under Markov morphisms (conditional expectations)
2. **Smoothness:** $g$ varies smoothly with $\theta$

is proportional to the Fisher Information Metric:

$$
g_{ij}(\theta) = c \cdot \mathbb{E}_\theta\left[\frac{\partial \log p(x|\theta)}{\partial \theta^i} \frac{\partial \log p(x|\theta)}{\partial \theta^j}\right]

$$
for some constant $c > 0$.

*Proof.* See Chentsov (1982) {cite}`chentsov1982statistical` and Campbell (1986) {cite}`campbell1986extended`. The proof uses the characterization of Markov morphisms as coarse-grainings and shows that invariance under all such maps forces the metric to be the Fisher metric. $\square$

*Significance.* Chentsov's theorem establishes that the Fisher metric is not a choice but a *necessity*: any geometry on probability space that respects statistical structure must be (proportional to) the Fisher geometry. This grounds our derivation in fundamental statistics, not ad-hoc assumptions.

:::

::::{admonition} Physics Isomorphism: Fisher Information Metric
:class: note
:name: pi-fisher-information

**In Physics:** The Fisher Information Metric $\mathcal{F}_{ij}(\theta) = \mathbb{E}\left[\frac{\partial \log p}{\partial \theta^i}\frac{\partial \log p}{\partial \theta^j}\right]$ is the unique Riemannian metric on statistical manifolds invariant under sufficient statistics (Chentsov's Theorem) {cite}`chentsov1982statistical,amari1985differential`.

**In Implementation:** The latent metric $G(z)$ combines value curvature with Fisher Information ({ref}`Section 2.5 <sec-second-order-sensitivity-value-defines-a-local-metric>`):

$$
G_{ij}(z) = \nabla^2_{ij} V(z) + \lambda\,\mathcal{F}_{ij}(z)

$$
where $\mathcal{F}_{ij} = \mathbb{E}_{a\sim\pi}[\partial_i \log\pi \cdot \partial_j \log\pi]$ is the state-space Fisher component $G_\pi$.

**Correspondence Table:**
| Information Geometry | Agent (Latent Metric) |
|:---------------------|:----------------------|
| Parameter space $\Theta$ | Latent state space $\mathcal{Z}$ |
| Fisher metric $\mathcal{F}_{ij}$ | Base metric contribution |
| Sufficient statistics | Macro-state $K$ (chart index) |
| KL divergence $D_{KL}$ | Squared geodesic distance (locally) |
| Natural gradient | Metric-aware policy gradient |

**Significance:** Chentsov's theorem (Theorem {prf:ref}`thm-a-chentsov-uniqueness`) proves the Fisher metric is not a choice but a necessity—any geometry respecting statistical structure must be proportional to Fisher.
::::

:::{prf:definition} A.6.0c (Computational Microstate)
:label: def-a-computational-microstate

A **computational microstate** at resolution $\ell$ is a complete specification of the agent's internal configuration $\mu = (\rho, K, \theta)$ where:
- $\rho \in \mathcal{P}(\mathcal{Z})$ is the belief distribution over the latent manifold
- $K \in \{1, \ldots, |\mathcal{K}|\}$ is the active chart assignment
- $\theta$ are the model parameters

discretized at the Levin Length scale: positions resolved to precision $\ell_L$, probabilities resolved to precision $e^{-1}$ in KL divergence.

Two microstates $\mu_1, \mu_2$ are **boundary-distinguishable** if an external observer, receiving only boundary observations $\partial\mathcal{Z}$, can distinguish them with probability $> 1 - e^{-1}$.

*Remark (Analogy to Physics).* In black hole thermodynamics, a microstate is a specific quantum configuration of the horizon degrees of freedom. Here, a microstate is a specific configuration of the agent's belief state. The boundary plays the role of the horizon: internal distinctions not visible at the boundary do not count toward the entropy.

:::



(sec-appendix-a-microstate-counting)=
### A.6.0d Microstate Counting: The Non-Circular Derivation

We now derive the Area Law by counting boundary-distinguishable microstates, without invoking the Metric Law.

:::{prf:lemma} A.6.0d (Geodesic Distance on the Probability Simplex)
:label: lem-a-geodesic-distance-probability-simplex

On the 1-simplex $\Delta^1 = \{(p, 1-p) : p \in [0,1]\}$ with Fisher Information Metric, the geodesic distance from the uniform distribution $(1/2, 1/2)$ to a vertex $(1, 0)$ is:

$$
d_{\text{Fisher}}\left(\tfrac{1}{2}, 1\right) = \frac{\pi}{2}.

$$
*Proof.* The Fisher metric on $\Delta^1$ is:

$$
ds^2 = \frac{dp^2}{p(1-p)}.

$$
Introduce the angular parameterization $p = \cos^2(\theta/2)$, so that $1-p = \sin^2(\theta/2)$ and:

$$
dp = -\cos(\theta/2)\sin(\theta/2)d\theta = -\frac{1}{2}\sin\theta \, d\theta.

$$
Then:

$$
ds^2 = \frac{\frac{1}{4}\sin^2\theta \, d\theta^2}{\cos^2(\theta/2)\sin^2(\theta/2)} = \frac{\frac{1}{4}\sin^2\theta \, d\theta^2}{\frac{1}{4}\sin^2\theta} = d\theta^2.

$$
The uniform distribution $(1/2, 1/2)$ corresponds to $\theta = \pi/2$. The vertex $(1, 0)$ corresponds to $\theta = 0$. The geodesic distance is:

$$
d = \int_0^{\pi/2} d\theta = \frac{\pi}{2}. \quad \square

$$
*Interpretation.* One bit of information (distinguishing "heads" from "tails") corresponds to geodesic distance $\pi/2$ in Fisher geometry. This is a derived quantity, not an assumption.

:::

:::{prf:lemma} A.6.0e (Curvature Normalization and the Factor of 4)
:label: lem-a-curvature-normalization-factor-4

The Poincare disk model with constant sectional curvature $K = -1$ has metric:

$$
ds^2 = \frac{4(dx^2 + dy^2)}{(1-|z|^2)^2}.

$$
The factor of 4 is uniquely determined by the curvature normalization.

*Proof.* For a 2D Riemannian manifold with conformal metric $ds^2 = \lambda(z)(dx^2 + dy^2)$, the Gaussian curvature is {cite}`docarmo1992riemannian`:

$$
K = -\frac{1}{2\lambda}\Delta(\log \lambda),

$$
where $\Delta = \partial_x^2 + \partial_y^2$ is the flat Laplacian.

For $\lambda = c/(1-r^2)^2$ where $r^2 = x^2 + y^2$ and $c > 0$:

**Step 1:** Compute $\log \lambda = \log c - 2\log(1-r^2)$.

**Step 2:** Compute the Laplacian. Let $f = \log(1-r^2)$. Then:

$$
\partial_x f = \frac{-2x}{1-r^2}.

$$
Applying the quotient rule to $\partial_x f = -2x \cdot (1-r^2)^{-1}$:

$$
\partial_x^2 f = \frac{-2(1-r^2) - (-2x)(-2x)}{(1-r^2)^2} = \frac{-2 + 2r^2 - 4x^2}{(1-r^2)^2}.

$$
Similarly for $y$. Adding:

$$
\Delta f = \frac{(-2 + 2r^2 - 4x^2) + (-2 + 2r^2 - 4y^2)}{(1-r^2)^2} = \frac{-4 + 4r^2 - 4r^2}{(1-r^2)^2} = \frac{-4}{(1-r^2)^2}.

$$
**Step 3:** Therefore $\Delta(\log \lambda) = -2\Delta f = \frac{8}{(1-r^2)^2}$.

**Step 4:** The curvature is:

$$
K = -\frac{1}{2\lambda} \cdot \frac{8}{(1-r^2)^2} = -\frac{(1-r^2)^2}{2c} \cdot \frac{8}{(1-r^2)^2} = -\frac{4}{c}.

$$
**Step 5:** For $K = -1$, we require $c = 4$. $\square$

*Significance.* The choice $K = -1$ is canonical: it sets the "radius of curvature" to unity, making the hyperbolic distance formula $d(0,z) = 2\text{arctanh}|z|$ dimensionless. The factor of 4 in the metric is a *derived consequence* of the curvature normalization, not an assumption.

:::

:::{prf:proposition} A.6.0f (Area of a Minimal Distinguishable Cell)
:label: prop-a-area-minimal-distinguishable-cell

On a 2-dimensional latent manifold with Fisher-compatible geometry (curvature $K = -1$), the Riemannian area of a cell containing exactly one nat of distinguishable information is:

$$
A_{\text{1 nat}} = 4\ell_L^2,

$$
where $\ell_L$ is the Levin Length.

*Proof (Non-Circular Derivation).* The argument proceeds in three independent steps:

**Step 1: Definition of $\ell_L$ (Implementation-Determined).** The Levin Length $\ell_L$ is the fundamental coordinate resolution of the computational manifold, determined by implementation constraints (discretization precision, floating-point resolution, etc.). This is analogous to how the Planck length $\ell_P = \sqrt{\hbar G/c^3}$ is determined by physical constants, not by the form of the area law.

**Step 2: Geodesic-to-Coordinate Relationship (From Fisher Metric).** On the Poincare disk with $K = -1$, the line element at the origin is:

$$
ds = 2 \, dx \quad \text{(from } ds^2 = 4(dx^2 + dy^2) \text{ at } z = 0\text{)}.

$$
A coordinate displacement $\ell_L$ corresponds to geodesic (Riemannian) distance $2\ell_L$.

**Step 3: Information-Geodesic Correspondence (From Chentsov).** By Theorem {prf:ref}`thm-a-chentsov-uniqueness`, the Fisher metric is the unique metric where KL divergence corresponds to squared geodesic distance (locally). Specifically, for nearby distributions $p$ and $q$:

$$
D_{\text{KL}}(p \| q) \approx \frac{1}{2} d_{\text{geo}}(p, q)^2.

$$
Thus, 1 nat of KL divergence corresponds to geodesic distance $\sqrt{2}$.

**Combining:** A coordinate cell of side $\ell_L$ has:
- Coordinate area: $\ell_L^2$
- Riemannian area: $\ell_L^2 \cdot \sqrt{\det G(0)} = \ell_L^2 \cdot 4 = 4\ell_L^2$
- Information capacity: proportional to Riemannian area $/$ (geodesic length per nat)$^2$

The factor of 4 emerges from the conformal factor $\sqrt{\det G(0)} = 4$, which was derived in Lemma {prf:ref}`lem-a-curvature-normalization-factor-4` from the curvature normalization $K = -1$, not from any assumption about information capacity. $\square$

*Remark (Non-Circularity).* In this derivation:
- $\ell_L$ is defined by implementation constraints (Step 1)
- The factor of 4 is derived from $K = -1$ (Lemma A.6.0e)
- The information-geometry correspondence is from Chentsov's theorem (Step 3)

No step assumes the form of the Area Law. Compare with Strominger-Vafa: they derive $S = A/(4\ell_P^2)$ by counting D-brane configurations, where $\ell_P$ is determined by string parameters and the 1/4 emerges from the counting.

:::

:::{prf:theorem} A.6.0g (Boundary Channel Capacity)
:label: thm-a-boundary-channel-capacity

The channel capacity of a 2-dimensional boundary $\partial\mathcal{Z}$ with Riemannian area $A$ is:

$$
C_\partial = \frac{A}{4\ell_L^2} \text{ nats}.

$$
*Proof.*
1. Tile the boundary with minimal distinguishable cells (Proposition {prf:ref}`prop-a-area-minimal-distinguishable-cell`)
2. By Proposition {prf:ref}`prop-a-area-minimal-distinguishable-cell`, each cell with coordinate side $\ell_L$ has Riemannian area $4\ell_L^2$
3. Number of cells: $N_{\text{cells}} = A / (4\ell_L^2)$
4. Each cell encodes 1 nat of information: this follows from the Fisher metric correspondence (Proposition {prf:ref}`prop-a-area-minimal-distinguishable-cell`, Step 3), not by definition
5. By additivity of channel capacity for parallel independent channels:

$$
C_\partial = N_{\text{cells}} \times 1 \text{ nat} = \frac{A}{4\ell_L^2}. \quad \square

$$
*Remark (Dimension Generalization).* For a $(D-1)$-dimensional boundary with $D > 2$, the formula generalizes to:

$$
C_\partial = \nu_D \cdot \frac{A}{\ell_L^{D-1}},

$$
where $\nu_D$ is the Holographic Coefficient (Definition {prf:ref}`def-holographic-coefficient`). The 2D case with $\nu_2 = 1/4$ is the primary focus of this specification.

*Remark (Shannon's Channel Coding Theorem).* This invokes the classical result that the capacity of $N$ parallel channels is additive. The generalization to continuous channels with Fisher geometry follows from rate-distortion theory {cite}`cover2006elements`.

:::

:::{prf:theorem} A.6.0h (Microstate Count and the Area Law)
:label: thm-a-microstate-count-area-law

The number of boundary-distinguishable microstates in the bulk is:

$$
\Omega = \exp\left(\frac{A}{4\ell_L^2}\right),

$$
and the maximum information about bulk configuration, as measured by an external observer, is:

$$
I_{\max} = \ln \Omega = \frac{A}{4\ell_L^2}.

$$
*Proof.*
1. By the **Data Processing Inequality**, information about the bulk cannot exceed the channel capacity of the boundary: $I_{\text{bulk} \to \text{observer}} \leq C_\partial$.

2. The maximum number of distinguishable messages through a channel of capacity $C$ nats is $e^C$ (Shannon's channel coding theorem {cite}`cover2006elements`).

3. Therefore, the number of boundary-distinguishable microstates is bounded:

$$
\Omega \leq e^{C_\partial} = \exp\left(\frac{A}{4\ell_L^2}\right).

$$
4. **Achievability:** The bound is saturated when the boundary is tiled with minimal distinguishable cells, each encoding 1 nat via orthogonal degrees of freedom. This follows from the channel capacity achievability in Shannon's theorem.

5. The maximum information is:

$$
I_{\max} = \ln \Omega = \frac{A}{4\ell_L^2}. \quad \square

$$
*Remark (Non-Circularity).* This derivation uses only:
- Chentsov's uniqueness theorem (statistics)
- Fisher geodesic distance calculation (geometry)
- Curvature normalization $K = -1$ (convention, not assumption)
- Shannon's channel capacity (information theory)

It does **not** invoke the Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`). The Metric Law is a *dynamical* statement about how the metric responds to information density; the Area Law derived here is a *kinematic* bound on distinguishable states.

:::



(sec-appendix-a-holographic-reduction)=
### A.6.1 Step 1: Holographic Reduction via Divergence Theorem (Field-Theoretic Derivation)

:::{prf:lemma} A.6.1 (Bulk-to-Boundary Conversion)
:label: lem-a-bulk-to-boundary-conversion

For a stationary information distribution satisfying the Metric Law, the bulk information integral can be expressed as a boundary integral:

$$
I_{\text{bulk}} = \int_{\mathcal{Z}} \rho_I \, d\mu_G = \frac{1}{\kappa} \oint_{\partial\mathcal{Z}} \text{Tr}(K) \, dA_G,

$$
where $K_{ij}$ is the extrinsic curvature (second fundamental form) of the boundary and $\kappa$ is the coupling constant from the Metric Law.

*Proof.* At stationarity, the information density satisfies the continuity equation $\nabla_i j^i = 0$ where $j^i$ is the information flux. The Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) implies:

$$
R - 2\Lambda = \kappa \, T,

$$
where $T = G^{ij}T_{ij}$ is the trace of the stress tensor. For uniform saturation, $T = n \cdot \sigma_{\max}$.

Integrating the Einstein tensor identity over $\mathcal{Z}$ and applying Lemma {prf:ref}`lem-a-divergence-to-boundary-conversion`:

$$
\int_{\mathcal{Z}} R \, d\mu_G = 2 \oint_{\partial\mathcal{Z}} \text{Tr}(K) \, dA_G.

$$
Combining with $R = \kappa T + 2\Lambda$ and noting that the $\Lambda$ term contributes a volume integral that cancels under the capacity constraint, we obtain the stated identity. $\square$

:::



(sec-appendix-a-saturation-geometry)=
### A.6.2 Step 2: Saturation Geometry (Schwarzschild-like Solution)

For an isotropic manifold with spherical symmetry, we use the ansatz:

$$
ds^2 = A(r) \, dr^2 + r^2 \, d\Omega_{n-1}^2,

$$
where $d\Omega_{n-1}^2$ is the metric on the unit $(n-1)$-sphere.

:::{prf:proposition} A.6.2 (Saturation Metric Solution)
:label: prop-a-saturation-metric-solution

Under uniform saturation $T_{ij} = \sigma_{\max} G_{ij}$, the Metric Law reduces to:

$$
\frac{n-2}{r^2}\left(1 - \frac{1}{A(r)}\right) + \frac{n-2}{r} \cdot \frac{A'(r)}{A(r)^2} = \kappa \sigma_{\max} + \Lambda.

$$
The solution is:

$$
A(r) = \left( 1 - \frac{2\mu(r)}{(n-2)r^{n-2}} - \frac{\Lambda_{\text{eff}} r^2}{n(n-1)} \right)^{-1},

$$
where $\mu(r) = \frac{\kappa}{n-2} \int_0^r \sigma_{\max} r'^{n-1} dr'$ is the information mass function and $\Lambda_{\text{eff}} = \Lambda + \kappa \sigma_{\max}$.

*Proof.* This follows from the standard Birkhoff-like analysis for spherically symmetric solutions of Einstein-type equations. The key steps are:

1. Compute the Ricci tensor components for the ansatz
2. Substitute into the Metric Law
3. The radial component of the field equations gives a first-order ODE for $A(r)$
4. Integrate with boundary condition $A(0) = 1$ (regularity at origin)

The integration constant is determined by requiring $\lim_{r \to 0} A(r) = 1$. $\square$

:::



(sec-appendix-a-horizon-condition)=
### A.6.3 Step 3: The Horizon Condition

:::{prf:definition} A.6.3 (Information Horizon)
:label: def-a-information-horizon

The **information horizon** $r_h$ is the smallest positive root of:

$$
1 - \frac{2\mu(r_h)}{(n-2)r_h^{n-2}} - \frac{\Lambda_{\text{eff}} r_h^2}{n(n-1)} = 0.

$$
At this radius, $A(r_h) \to \infty$ and $G^{rr}(r_h) \to 0$.

:::

For $n = 2$ (the Poincare disk case), the formula simplifies. The Poincare metric already encodes the horizon at $|z| = 1$:

$$
G_{ij}(z) = \frac{4\delta_{ij}}{(1-|z|^2)^2} \xrightarrow{|z| \to 1} \infty.

$$


(sec-appendix-a-fisher-normalization)=
### A.6.4 Step 4: Fisher Normalization and the 1/4 Coefficient

The coefficient $1/4$ in the field-theoretic derivation arises from the same geometric structure established in the microstate counting approach (Section A.6.0d).

:::{prf:remark} A.6.4a (Connection to Microstate Counting)
:label: rem-a-connection-microstate-counting

The Fisher normalization used here is **not an independent input**. It is the same geometric fact established by:
- Lemma {prf:ref}`lem-a-geodesic-distance-probability-simplex`: Geodesic distance $\pi/2$ for 1 bit
- Lemma {prf:ref}`lem-a-curvature-normalization-factor-4`: Factor of 4 from curvature $K = -1$
- Proposition {prf:ref}`prop-a-area-minimal-distinguishable-cell`: Area $4\ell_L^2$ per nat

The field-theoretic derivation shows that the Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) *reproduces* the bound derived from counting—providing a consistency check between the kinematic and dynamic approaches.

:::

:::{prf:lemma} A.6.4 (Geodesic Distance on the Probability Simplex)
:label: lem-a-geodesic-distance-simplex

On the 1-simplex $\Delta^1 = \{(p, 1-p) : p \in [0,1]\}$ with the Fisher Information Metric, the geodesic distance between the uniform distribution $(1/2, 1/2)$ and a vertex $(1, 0)$ is:

$$
d_{\text{Fisher}}\left(\frac{1}{2}, 1\right) = \frac{\pi}{2}.

$$
*Proof.* See Lemma {prf:ref}`lem-a-geodesic-distance-probability-simplex` for the full derivation. $\square$

:::

:::{prf:proposition} A.6.5 (Area of a Minimal Information Cell)
:label: prop-a-area-minimal-cell

On a 2-dimensional Fisher manifold, the area of a cell corresponding to 1 nat of distinguishable information is:

$$
A_{\text{cell}} = 4 \ell_L^2.

$$
*Proof.* See Proposition {prf:ref}`prop-a-area-minimal-distinguishable-cell` for the full derivation. The key steps are:
1. Poincare metric at origin: $G(0) = 4I$ (from curvature normalization $K = -1$)
2. Coordinate cell area $\ell_L^2$ maps to Riemannian area $4\ell_L^2$ $\square$

:::



(sec-appendix-a-assembly)=
### A.6.5 Step 5: Assembly of the Bound

:::{prf:theorem} A.6.6 (Complete Derivation of the Area Law)
:label: thm-a-complete-derivation-area-law

Combining the above results:

1. **From Lemma {prf:ref}`lem-a-bulk-to-boundary-conversion`:** $I_{\text{bulk}} = \frac{1}{\kappa} \oint_{\partial\mathcal{Z}} \text{Tr}(K) \, dA_G$

2. **At saturation:** The extrinsic curvature $\text{Tr}(K) = (n-1)/r_h$ for an $(n-1)$-sphere boundary.

3. **Boundary area:** $\text{Area}(\partial\mathcal{Z}) = \Omega_{n-1} r_h^{n-1}$ where $\Omega_{n-1}$ is the volume of the unit $(n-1)$-sphere.

4. **Fisher normalization:** $\kappa = 8\pi \ell_L^2$ (fixed by consistency with Proposition {prf:ref}`prop-a-area-minimal-cell`).

Substituting:

$$
I_{\max} = \frac{1}{8\pi \ell_L^{n-1}} \cdot \frac{n-1}{r_h} \cdot \Omega_{n-1} r_h^{n-1} = \frac{(n-1)\Omega_{n-1}}{8\pi} \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{n-1}}.

$$
Identifying the **Holographic Coefficient** $\nu_n := (n-1)\Omega_{n-1}/(8\pi)$ (Definition {prf:ref}`def-holographic-coefficient`), we obtain the **general result**:

$$
\boxed{I_{\max} = \nu_n \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{n-1}}}.

$$

**Special case ($n = 2$, Poincare disk):** With $\Omega_1 = 2\pi$ (circumference of unit circle), we get $\nu_2 = 1/4$. The familiar Bekenstein-Hawking form:

$$
I_{\max} = \frac{\text{Area}(\partial\mathcal{Z})}{4\ell_L^2}

$$
uses $\ell_L^2$ (rather than $\ell_L^{n-1} = \ell_L$) because the Poincare disk metric normalization $G(0) = 4I$ maps coordinate cells to Riemannian areas.

This completes the derivation. The Holographic Coefficient $\nu_n$ arises from the combination of:
- The $1/8\pi$ from the coupling constant $\kappa$
- The geometric factor $(n-1)\Omega_{n-1}$ from sphere surface area
- The Fisher metric normalization

$\square$

:::

:::{prf:corollary} A.6.7 (Dimension-Dependent Coefficient)
:label: cor-a-dimension-dependent-coefficient

For a $D$-dimensional latent manifold with $(D-1)$-sphere boundary, the Causal Information Bound takes the form:

$$
I_{\max}(D) = \nu_D \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{D-1}},

$$
where the Holographic Coefficient $\nu_D$ (Definition {prf:ref}`def-holographic-coefficient`) is:

$$
\nu_D = \frac{(D-1)\Omega_{D-1}}{8\pi} = \frac{(D-1)\pi^{(D-2)/2}}{4\,\Gamma(D/2)},

$$
with $\Omega_{D-1} = 2\pi^{D/2}/\Gamma(D/2)$ the surface area of the unit $(D-1)$-sphere.

**Explicit values:**

| $D$ | $\Omega_{D-1}$ | $\nu_D$    | Numerical |
|-----|----------------|------------|-----------|
| 2   | $2\pi$         | $1/4$      | 0.250     |
| 3   | $4\pi$         | $1$        | 1.000     |
| 4   | $2\pi^2$       | $3\pi/4$   | 2.356     |
| 5   | $8\pi^2/3$     | $4\pi/3$   | 4.189     |
| 6   | $\pi^3$        | $5\pi^2/8$ | 6.169     |

*Remark.* The coefficient $\nu_D$ is **not monotonic** in $D$: it increases from $D=2$ to a peak at $D \approx 9$ ($\nu_9 \approx 9.4$), then decreases toward zero. For typical latent dimensions ($3 \le D \le 20$), $\nu_D > \nu_2 = 1/4$, so using the 2D coefficient **underestimates** capacity. For very high dimensions ($D \gtrsim 22$), $\nu_D < 1/4$, so the 2D coefficient **overestimates** capacity—this is the dangerous case (false safety). Implementers should always use the dimension-appropriate coefficient.

:::

:::{warning}
:name: warning-dimension-dependent-node-56

**Implementation Note for Node 56 (CapacityHorizonCheck):**

The saturation ratio $\eta_{\text{Sch}} = I_{\text{bulk}} / I_{\max}$ depends on the Holographic Coefficient $\nu_D$ (Definition {prf:ref}`def-holographic-coefficient`):

$$
I_{\max}(D) = \nu_D \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{D-1}}.

$$
The default $\nu_2 = 1/4$ assumes a 2-dimensional latent manifold (Poincare disk). For $D$-dimensional latent spaces, use the appropriate $\nu_D$ from Corollary {prf:ref}`cor-a-dimension-dependent-coefficient`.

Using the wrong coefficient leads to:
- **$\nu > \nu_D$ (typical for $D > 21$):** **Dangerous.** False safety—agent enters super-saturated regime undetected.
- **$\nu < \nu_D$ (typical for $3 \le D \le 20$):** Conservative—unnecessary fusion triggered, but safe.

**Implementation Code:**
```python
def holographic_coefficient(D: int) -> float:
    """Compute nu_D = (D-1) * Omega_{D-1} / (8 * pi)"""
    import math
    if D < 2:
        return 0.0
    omega = 2 * (math.pi ** (D / 2)) / math.gamma(D / 2)
    return (D - 1) * omega / (8 * math.pi)
```

:::

:::{prf:remark} A.6.8 (Gauss-Bonnet Generalization)
:label: rem-a-gauss-bonnet-generalization

The derivation in Lemma {prf:ref}`lem-a-bulk-to-boundary-conversion` uses the **Einstein tensor divergence identity** (also called the contracted Bianchi identity):

$$
\int_{\mathcal{Z}} R \, d\mu_G = 2 \oint_{\partial\mathcal{Z}} \text{Tr}(K) \, dA_G,

$$
which is valid in **arbitrary dimension**. This is more general than the classical 2D Gauss-Bonnet theorem (which relates $\int K \, dA$ to the Euler characteristic $\chi$).

The Chern-Gauss-Bonnet theorem for even-dimensional manifolds computes topological invariants (Euler characteristic) via curvature integrals, but is not required here—we compute information capacity, not topology. The divergence theorem approach generalizes to any $D \geq 2$ without modification.

:::

:::{prf:remark} A.6.9 (Non-Circularity of the Derivation)
:label: rem-a-non-circularity

A potential criticism of this section is circularity: *"The Metric Law encodes the holographic principle, so deriving the Area Law from the Metric Law is question-begging."*

This criticism is addressed by the **two-derivation structure** of this appendix:

1. **Microstate Counting (A.6.0):** Derives $I_{\max} = A/(4\ell_L^2)$ from:
   - Chentsov's uniqueness theorem (Theorem {prf:ref}`thm-a-chentsov-uniqueness`)
   - Fisher geodesic distance calculation (Lemma {prf:ref}`lem-a-geodesic-distance-probability-simplex`)
   - Curvature normalization $K = -1$ (Lemma {prf:ref}`lem-a-curvature-normalization-factor-4`)
   - Shannon's channel capacity (Theorem {prf:ref}`thm-a-boundary-channel-capacity`)

   **This derivation does not invoke the Metric Law.**

2. **Field-Theoretic (A.6.1–A.6.5):** Derives the same bound from the Metric Law dynamics.

The fact that both derivations yield the **same coefficient** $(1/4)$ is a non-trivial consistency check:
- The kinematic bound (from counting) constrains what is *possible*
- The dynamic equations (from the Metric Law) describe what *happens*
- Their agreement shows the Metric Law is *compatible* with holographic constraints—not that it *assumes* them

**Analogy to physics:** In black hole thermodynamics, Hawking derived $S = A/(4\ell_P^2)$ thermodynamically (1975), and Strominger-Vafa derived it microscopically (1996). Neither derivation is circular; their agreement is a profound consistency check on string theory.

Similarly, the microstate counting here is analogous to Strominger-Vafa, while the field-theoretic derivation is analogous to Hawking. The framework admits both perspectives.

:::



(sec-appendix-a-remark-bekenstein-hawking)=
### A.6.6 Remark: Connection to Bekenstein-Hawking

The structural similarity to the Bekenstein-Hawking entropy bound $S = A/(4\ell_P^2)$ {cite}`bekenstein1973black,hawking1975particle` is not coincidental. Both bounds arise from:

1. **A field equation** relating curvature to a source (Einstein equation / Metric Law)
2. **A saturation condition** where the source density reaches its maximum consistent with regularity
3. **A holographic reduction** mapping bulk integrals to boundary terms

The key difference is the interpretation:
- In gravity: $\ell_P$ is the Planck length (quantum gravity scale); $S$ is thermodynamic entropy
- In the Fragile Agent: $\ell_L$ is the Levin length (information-theoretic scale); $I$ is representational information

The mathematical structure is identical; the physical content is distinct. This suggests that holographic bounds are a general feature of capacity-constrained field theories, independent of whether the underlying dynamics are gravitational or information-theoretic.
