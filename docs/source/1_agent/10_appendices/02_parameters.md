(sec-appendix-b-units-parameters-and-coefficients)=
# {ref}`Appendix B <sec-appendix-b-units-parameters-and-coefficients>`: Units, Parameters, and Coefficients (Audit Table)

## TLDR

- Centralize **units, parameters, and coefficients** so thresholds and regularizers are auditable across chapters.
- Use this when implementing: it prevents silent unit mismatches and makes “what is this constant?” a lookup, not a hunt.

(sec-appendix-b-base-units)=
## B.1 Base Units (Information + Steps)

We use a purely information-theoretic unit system:
- **Information / cost:** nats ($\mathrm{nat}$).
- **Interaction time $t$:** discrete environment steps ($\mathrm{step}$).
- **Computation time $s$:** internal solver time (continuous; normalized units unless mapped to wall-clock).
- **Scale time $\tau$:** depth coordinate (dimensionless).
- **Memory time $t'$:** discrete past index ($\mathrm{step}$, with $t' < t$).

Conventions:
- Entropies $H(\cdot)$, mutual information $I(\cdot;\cdot)$, and divergences $D_{\mathrm{KL}}$ are measured in $\mathrm{nat}$.
- Value/cost scalars ($V$, $F_t$, budgets, thresholds) are measured in $\mathrm{nat}$.
- Per-step rates (HJB terms, $\Delta V$, any “cost rate”) are measured in $\mathrm{nat/step}$ (interaction time).
- Latent coordinates ($z$, $z_n$, $z_{\mathrm{tex}}$, code embeddings $e_k$) are treated as **normalized/dimensionless**; any physical units should be absorbed into preprocessing and encoder normalization.

(sec-appendix-b-parameter-coefficient-units)=
## B.2 Parameter / Coefficient Units (by Role)

| Symbol                                                                               | Meaning (context)                                                                                   | Units                                          |
|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------|
| $t$                                                                                  | interaction step index                                                                              | $\mathrm{step}$                                |
| $t'$                                                                                 | memory time index ($t' < t$)                                                                        | $\mathrm{step}$                                |
| $s$                                                                                  | computation time (internal solver)                                                                  | solver-time units (normalized)                 |
| $\tau$                                                                               | scale time (depth)                                                                                  | dimensionless                                  |
| $\Delta t$                                                                           | optional mapping from steps to wall-clock                                                           | $\mathrm{s/step}$                              |
| $r_t$                                                                                | reward/cost per step                                                                                | $\mathrm{nat}$                                 |
| $\mathcal{R}$                                                                        | reward 1-form (rate per length; paired with velocity gives nat/step)                                | $\mathrm{nat}/[\text{length}]$                 |
| $A$                                                                                  | non-conservative component of the reward 1-form ($\mathcal{R}=d\Phi + A$)                            | $\mathrm{nat}/[\text{length}]$                 |
| $V$                                                                                  | value / cost-to-go                                                                                  | $\mathrm{nat}$                                 |
| $\Delta V$                                                                           | value change per step                                                                               | $\mathrm{nat/step}$                            |
| $\mathfrak{D}$                                                                       | control-effort / regularization rate term                                                           | $\mathrm{nat/step}$                            |
| $\lambda$                                                                            | Lyapunov rate in $\dot V\le -\lambda V$ (continuous-time form)                                      | $s^{-1}$                                       |
| $\gamma$                                                                             | discount factor (MaxEnt RL)                                                                         | dimensionless                                  |
| $H$                                                                                  | horizon / planning depth                                                                            | $\mathrm{step}$                                |
| $T_c$                                                                                | cognitive temperature / entropy-regularization coefficient ({prf:ref}`def-cognitive-temperature`)  | dimensionless                                  |
| $\beta_{\text{ent}}$                                                                 | exponential-family scale in $\exp(-\beta_{\text{ent}} V)$                                            | dimensionless                                  |
| $\Theta$                                                                             | local conditioning proxy (Definition {prf:ref}`def-local-conditioning-scale`)                       | dimensionless (when $z$ normalized)            |
| $\beta_{\text{cpl}}$                                                                 | coupling coefficient from $\Theta$ (Definition {prf:ref}`def-local-conditioning-scale`)             | $\mathrm{nat}/[z]^2$                           |
| $\epsilon$                                                                           | numeric stabilizer / threshold                                                                      | inherits compared quantity                     |
| $\eta$                                                                               | step size / learning-rate symbol                                                                    | dimensionless (in normalized coordinates)      |
| $\beta$                                                                              | VQ-VAE commitment weight ({ref}`Section 2.2b <sec-the-shutter-as-a-vq-vae>` / 3.3)                                                       | dimensionless                                  |
| $\beta_n$                                                                            | nuisance KL weight (structured residual)                                                            | dimensionless                                  |
| $\beta_{\mathrm{tex}}$                                                               | texture KL weight (reconstruction-only residual)                                                    | dimensionless                                  |
| $\beta_K$                                                                            | macro codelength weight (rate term)                                                                 | dimensionless                                  |
| $\lambda_{\text{use}}$                                                               | codebook usage regularizer weight                                                                   | dimensionless                                  |
| $\lambda_{\text{*}}$                                                                 | composite-loss weights (e.g. $\lambda_{\text{shutter}},\lambda_{\text{ent}},\lambda_{\text{zeno}}$) | dimensionless                                  |
| $\lambda,\mu,\nu$                                                                    | VICReg component weights                                                                            | dimensionless                                  |
| $\alpha,\beta_{\pi},\gamma,\delta$                                                   | scaling coefficients ({ref}`Section 3.2 <sec-scaling-exponents-characterizing-the-agent>`)                                                                  | dimensionless                                  |
| $V_{\text{max}},V_{\text{limit}},V_{\text{proxy}},V_{\text{true}},B_{\text{switch}}$ | risk/cost budgets and thresholds                                                                    | $\mathrm{nat}$                                 |
| $\lambda_{\text{in}},\lambda_{\text{mix}}$                                           | grounding/mixing information rates                                                                  | $\mathrm{nat/step}$                            |
| $q_{k\to k'}$                                                                        | macro transition probabilities                                                                      | dimensionless                                  |
| $C_{\partial}$                                                                       | boundary information capacity                                                                       | $\mathrm{nat}$                                 |
| $I_{\text{bulk}}$                                                                    | bulk information volume                                                                             | $\mathrm{nat}$                                 |
| $\ell$                                                                               | boundary resolution scale                                                                           | boundary-length units (chosen)                 |
| $\eta_\ell$                                                                          | boundary area-per-nat at resolution $\ell$                                                          | $[dA_G]/\mathrm{nat}$                          |
| $\Lambda$                                                                            | curvature/capacity offset constant (metric law)                                                     | $[z]^{-2}$                                     |
| $\kappa$                                                                             | coupling in $R_{ij}-\\tfrac12R G_{ij}+\\Lambda G_{ij}=\\kappa T_{ij}$                               | chosen so $\kappa T_{ij}$ has units $[z]^{-2}$ |
| $U(z)$                                                                               | hyperbolic information potential $-d_{\mathbb{D}}(0,z)$ ({ref}`Section 21.1 <sec-hyperbolic-volume-and-entropic-drift>`)                              | $\mathrm{nat}$                                 |
| $T_c(\tau)$                                                                          | generative temperature schedule ({ref}`Section 21.2 <sec-policy-control-field>`)                                                      | dimensionless                                  |
| $\kappa_T$                                                                           | temperature annealing rate ({ref}`Section 21.2 <sec-policy-control-field>`)                                                           | $\tau^{-1}$                                    |
| $\phi_c$                                                                             | Möbius automorphism moving $c$ to origin ({ref}`Section 21.3 <sec-the-retrieval-texture-firewall>`)                                             | dimensionless (isometry)                       |
| $\lambda(z)$                                                                         | conformal factor $2/(1-\lvert z\rvert^2)$ ({ref}`Section 21.4 <sec-summary-and-diagnostic-node>`)                                            | $[z]^{-1}$                                     |
| $\sigma_{\text{tex}}$                                                                | base texture standard deviation ({ref}`Section 21.4 <sec-summary-and-diagnostic-node>`)                                                      | $[z_{\text{tex}}]$                             |
| $R_{\text{cutoff}}$                                                                  | geometric stopping radius ({ref}`Section 21.3 <sec-bulk-boundary-independence>`)                                                            | dimensionless                                  |
| $\epsilon_{\text{conv}}$                                                             | convergence stopping threshold ({ref}`Section 21.3 <sec-bulk-boundary-independence>`)                                                       | $\mathrm{nat}/\tau$                            |
| $S_{\mathrm{OM}}$                                                                    | Onsager-Machlup stochastic action ({ref}`Section 22.1 <sec-the-stochastic-action-principle>`)                                                    | $\mathrm{nat}$                                 |
| $\Phi_{\text{gen}}$                                                                  | generative potential $\alpha U + (1-\alpha)V_{\text{critic}}$ ({ref}`Section 22.3 <sec-the-unified-effective-potential>`)                        | $\mathrm{nat}$                                 |
| $\alpha$                                                                             | generation-control interpolation parameter ({ref}`Section 22.3 <sec-the-unified-effective-potential>`)                                           | dimensionless                                  |
| $\gamma$                                                                             | friction coefficient in overdamped limit ({ref}`Section 22.4 <sec-the-geodesic-baoab-integrator>`)                                             | $s^{-1}$                                       |
| $\beta_{\text{curl}}$                                                                | curl/Lorentz coupling coefficient ({ref}`Section 22.1 <sec-the-stochastic-action-principle>`)                                                     | dimensionless                                  |
| $\lambda_{\text{jump}}$                                                              | Poisson jump intensity ({ref}`Section 22.2 <sec-the-coupled-jump-diffusion-sde>`)                                                               | $s^{-1}$                                       |
| $\eta$                                                                               | multiplicative jump factor for mass ({ref}`Section 22.2 <sec-the-coupled-jump-diffusion-sde>`)                                                  | dimensionless                                  |
| $\mathcal{A}_y$                                                                      | sub-atlas for class $y$ ({ref}`Section 25.1 <sec-relationship-to-the-context-conditioned-framework>`)                                                              | —                                              |
| $V_y$                                                                                | class-conditioned potential ({ref}`Section 25.2 <sec-the-semantic-potential>`)                                                          | $\mathrm{nat}$                                 |
| $\beta_{\text{class}}$                                                               | class temperature (inverse semantic diffusion) ({ref}`Section 25.2 <sec-the-semantic-potential>`)                                       | dimensionless                                  |
| $\mathcal{B}_y$                                                                      | attractor basin for class $y$ ({ref}`Section 25.2 <sec-the-semantic-potential>`)                                                        | —                                              |
| $\gamma_{\text{sep}}$                                                                | class separation strength ({ref}`Section 25.3 <sec-metric-segmentation-via-jump-rate-modulation>`)                                                            | dimensionless                                  |
| $\lambda_{i\to j}^{\text{sup}}$                                                      | class-modulated jump rate ({ref}`Section 25.3 <sec-metric-segmentation-via-jump-rate-modulation>`)                                                            | $s^{-1}$                                       |
| $\mathcal{L}_{\text{purity}}$                                                        | chart purity loss $H(Y\mid K)$ ({ref}`Section 25.4 <sec-the-supervised-topology-loss>`)                                                       | $\mathrm{nat}$                                 |
| $\mathcal{L}_{\text{balance}}$                                                       | load balance loss ({ref}`Section 25.4 <sec-the-supervised-topology-loss>`)                                                                    | $\mathrm{nat}$                                 |
| $\mathcal{L}_{\text{metric}}$                                                        | metric contrastive loss ({ref}`Section 25.4 <sec-the-supervised-topology-loss>`)                                                              | $\mathrm{nat}$                                 |
| $\mathcal{L}_{\text{route}}$                                                         | route alignment loss ({ref}`Section 25.4 <sec-the-supervised-topology-loss>`)                                                                 | $\mathrm{nat}$                                 |
| $\epsilon_{\text{purity}}$                                                           | purity threshold ({ref}`Section 25.1 <sec-relationship-to-the-context-conditioned-framework>`)                                                                     | dimensionless                                  |

(sec-appendix-b-symbol-overload)=
## B.3 Symbol Overload (Important)

Some Greek letters are intentionally overloaded in different submodels:
- $\beta_{\text{ent}}$, $\beta_{\text{cpl}}$, and $\beta_{\text{curl}}$ are distinct subscripts for entropy, coupling, and curl terms; plain $\beta$ is reserved for the VQ-VAE commitment weight and some contrastive-loss scales where noted.
- $\tau$ appears as (i) scale time ({ref}`Section 1.3 <sec-the-chronology-temporal-distinctions>`), (ii) the entropy-weight coefficient in Section 2.11.3, and (iii) the temperature in some contrastive losses; use local definitions.
- $\gamma$ appears as (i) discount factor, (ii) the World Model volatility scaling coefficient $\gamma$ ({ref}`Section 3.2 <sec-scaling-exponents-characterizing-the-agent>`), and (iii) friction coefficient in overdamped dynamics ({ref}`Section 22.4 <sec-the-geodesic-baoab-integrator>`).
- $\lambda$ appears as (i) Lyapunov rate ($s^{-1}$), (ii) generic loss weights (dimensionless), and (iii) other Lagrange multipliers (units stated locally).
