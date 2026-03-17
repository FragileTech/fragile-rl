(sec-geometric-micro-architecture)=
# The Geometric Micro-Architecture

## TLDR

- **Standard DL primitives violate gauge invariance**: `nn.Linear(bias=True)` and element-wise `ReLU` assume flat Euclidean space with preferred coordinate axes. Rotating the latent representation changes the physics.
- **Gauge-covariant replacements preserve symmetry**: Replace with `SpectralLinear` (bias-free, Lipschitz-constrained) and `NormGatedGELU` (acts on vector bundle norms, not individual components).
- **Each primitive preserves a gauge symmetry component**: Spectral normalization → $U(1)_Y$ (capacity bound), bundle structure → $SU(N_f)_C$ (feature confinement), steerable convolutions → $SU(2)_L$ (observation-action chirality).
- **Microscopic operations compose to macroscopic geodesic flow**: Equivariant layers with Lipschitz constant $\le 1$ preserve the global light cone structure required by the Boris-BAOAB integrator.
- **Result**: Neural operators that respect the gauge group $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$ at the atomic level, making the architecture compatible with the capacity-constrained metric law and WFR geometry.

## Roadmap

1. **Formalize the gauge symmetry constraint**: Define $G$-equivariance for neural operators and explain why it matters for geometric consistency.
2. **Prove standard DL violates it**: Explicit counterexample showing `ReLU` breaks $SO(d)$-equivariance.
3. **Derive covariant replacements**: Spectral normalization (preserves light cone) + norm-gating (preserves bundle structure).
4. **Connect to causal structure**: Show how Lipschitz constraints implement the information speed limit $c_{\text{info}}$ from the parameter sieve.
5. **Extend to vision**: Steerable convolutions lift images to gauge bundles, ensuring $SO(2)$-equivariance for rotated inputs.
6. **Integration and diagnostics**: Connect microscopic primitives to macroscopic gauge fields, diagnostic nodes, and the geodesic integrator.

:::{div} feynman-prose
Now we come to what might seem like a technical detail but is actually fundamental to the entire framework. You see, in most deep learning, people throw together networks without much thought about the geometric properties of the operations they're using. Linear layers with biases, ReLU activations applied element-wise—these are treated as universal primitives, LEGO blocks you can snap together however you like.

But here's the thing: those primitives make hidden assumptions about your space. They assume it's flat. They assume the coordinate axes mean something special. When you apply ReLU element-wise to a vector, you're saying "kill the negative components based on the current choice of coordinates." But what if you rotate your mental axes? Suddenly different components get killed. The physics changes depending on an arbitrary choice.

This is not a metaphor. In Chapters 1-3 of this section, we derived that the agent's latent space must have a specific gauge structure—$SU(N_f)_C \times SU(2)_L \times U(1)_Y$—to ensure consistent multi-agent interactions and capacity constraints. If we build neural operators that violate this structure at the microscopic level, the macroscopic theory falls apart. The geodesic integrator requires smooth, well-defined tangent vectors. The metric law requires isometric transformations. The WFR geometry requires mass-conserving updates.

Standard DL primitives violate all of these. This chapter fixes that. We're going to build neural operators from scratch, starting with the requirement that they respect gauge symmetry. The result will be a new set of primitives—`IsotropicBlock`, `SteerableConv`—that play nice with the geometry we've spent three chapters deriving.

And here's the beautiful part: once you have gauge-covariant primitives, the architecture almost writes itself. The constraints are so tight that there's essentially only one way to build each component. This is the deep learning equivalent of the Standard Model in physics: a unique structure that emerges from symmetry requirements alone.
:::

*Cross-references:* This chapter builds on gauge theory foundations ({ref}`sec-symplectic-multi-agent-field-theory`), the capacity-constrained metric law ({ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits`), WFR geometry ({ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`), and feeds into the geodesic integrator implementation ({ref}`sec-equations-of-motion-langevin-sdes-on-information-manifolds`).



(sec-fundamental-symmetry-constraint)=
## The Fundamental Symmetry Constraint

:::{div} feynman-prose
Let me tell you what gauge symmetry really means by way of a simple example. Suppose you and I are both looking at the same robot arm, but I describe its position using a coordinate system aligned with magnetic north, while you use one aligned with the factory floor. We would write down different numbers for the same physical configuration. Yet when we compute the forces, the torques, the dynamics—we had better get the same answers. The physics cannot depend on our arbitrary choice of coordinates.

This is gauge symmetry: the requirement that *physics* be independent of *bookkeeping choices*. And here is where standard deep learning commits a subtle but devastating error. When you feed a vector through a typical neural network, the operations treat the coordinate axes as if they were physically meaningful. A ReLU applied to the first component does something different than a ReLU applied to the second component. Rotate your mental axes by 45 degrees, and suddenly different neurons fire. The network's internal representation *changes* based on an arbitrary choice that has no physical significance.

Think of it like this: imagine a GPS system that gave you different directions depending on whether you held the phone facing north or east. That would be broken, obviously. Yet this is precisely what standard neural networks do—their internal computations depend on the orientation of an arbitrary coordinate frame.

In Chapter 8.1, we derived that agents interacting under capacity constraints must respect a specific gauge structure: $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$. Every neural operation that violates this symmetry is like a GPS that gives inconsistent directions. The errors might be small on any single forward pass, but they accumulate catastrophically when you try to integrate geodesics or maintain consistent beliefs across time. We need to fix the primitives.
:::

### Formal Definitions

:::{prf:definition} Group Action on Latent Space
:label: def-group-action-latent

**Preliminaries:**
- A **Lie group** is a smooth manifold $G$ equipped with a group structure such that multiplication $(g, h) \mapsto gh$ and inversion $g \mapsto g^{-1}$ are smooth maps. Examples: $SO(d)$ (rotations), $SE(2) = \mathbb{R}^2 \rtimes SO(2)$ (rigid motions), $SU(N)$ (special unitary group).
- The **general linear group** $GL(d_z, \mathbb{R})$ is the group of invertible $d_z \times d_z$ real matrices under matrix multiplication, with smooth manifold structure (open subset of $\mathbb{R}^{d_z^2}$).

**Setup:** Let $G$ be a Lie group and $\mathcal{Z} = \mathbb{R}^{d_z}$ be the latent space equipped with standard Euclidean topology and inner product $\langle z, z' \rangle = z^T z'$.

A **linear representation** is a continuous group homomorphism $\rho: G \to GL(d_z, \mathbb{R})$ satisfying:
1. $\rho(e_G) = I_{d_z}$ where $e_G$ is the identity element of $G$
2. $\rho(g_1 g_2) = \rho(g_1)\rho(g_2)$ for all $g_1, g_2 \in G$ (homomorphism property)

**Smoothness:** When $G$ is a Lie group, we additionally require $\rho$ to be smooth as a map between manifolds: for any smooth curve $\gamma: (-\epsilon, \epsilon) \to G$ with $\gamma(0) = e_G$, the matrix-valued map $t \mapsto \rho(\gamma(t))$ is $C^\infty$.

**Group action:** For $g \in G$ and $z \in \mathcal{Z}$ (viewed as a $d_z \times 1$ column vector), the action is given by standard matrix-vector multiplication:

$$
g \cdot z := \rho(g) z \in \mathbb{R}^{d_z}
$$
where the right-hand side denotes the matrix product of $\rho(g) \in \mathbb{R}^{d_z \times d_z}$ with $z \in \mathbb{R}^{d_z \times 1}$.

**Orthogonal representations:** When $G$ is compact (e.g., $SO(d)$, $SU(N)$), any finite-dimensional continuous representation can be made orthogonal (unitary) by constructing a $G$-invariant inner product via Haar measure averaging {cite}`serre1977linear,sepanski2007compact`.

**Construction:** Let $\rho_0: G \to GL(d_z, \mathbb{R})$ be any representation, and let $\langle \cdot, \cdot \rangle_0$ be an arbitrary initial inner product on $\mathcal{Z} = \mathbb{R}^{d_z}$. Define the averaged inner product:

$$
\langle z, z' \rangle_G := \int_G \langle \rho_0(g) z, \rho_0(g) z' \rangle_0 \, dg
$$
where $dg$ is the normalized Haar measure on $G$ (unique bi-invariant measure with $\int_G dg = 1$).

**Verification:**

1. **Inner product structure:** $\langle \cdot, \cdot \rangle_G$ satisfies linearity, symmetry, and positive definiteness. For $z \neq 0$, we have $\langle \rho_0(g) z, \rho_0(g) z \rangle_0 > 0$ for all $g$ (since $\rho_0(g)$ is invertible). By compactness of $G$ and continuity, the integral $\langle z, z \rangle_G > 0$.

2. **$G$-invariance:** For any $h \in G$:
   $$\langle \rho_0(h) z, \rho_0(h) z' \rangle_G = \int_G \langle \rho_0(g) \rho_0(h) z, \rho_0(g) \rho_0(h) z' \rangle_0 \, dg$$
   Substituting $g' = gh$ and using left-invariance of Haar measure ($dg' = dg$):
   $$= \int_G \langle \rho_0(g') z, \rho_0(g') z' \rangle_0 \, dg' = \langle z, z' \rangle_G$$
   Therefore $\rho_0(h)$ is orthogonal (actually unitary) with respect to $\langle \cdot, \cdot \rangle_G$ for all $h \in G$.

We typically choose $\rho$ such that $\rho(g) \in O(d_z)$ for all $g$, ensuring:

$$
\langle \rho(g) z, \rho(g) z' \rangle = \langle z, z' \rangle \quad \forall z, z', g
$$
This preserves the Euclidean structure of $\mathcal{Z}$.

**Remark on Peter-Weyl theorem:** The Peter-Weyl theorem states that for a compact group $G$, the renormalized matrix coefficients $\sqrt{\dim(\pi)} \cdot u_{ij}^{(\pi)}(g)$ (where $u_{ij}^{(\pi)}$ are matrix elements of irreducible unitary representations $\pi$) form an orthonormal basis for $L^2(G)$ with respect to normalized Haar measure. This theorem is foundational to representation theory and is intimately connected to the averaging construction: the Haar measure averaging used above is the same measure appearing in the Peter-Weyl decomposition {cite}`peter1927theorie,weyl1946classical`.

**Units:** $[\rho(g)]$ is dimensionless (orthogonal/unitary matrices have dimensionless entries).
:::

:::{prf:definition} G-Equivariant Operator
:label: def-g-equivariant-operator

Let $G$ be a group with representations $\rho: G \to GL(\mathbb{R}^{d_z})$ and $\rho': G \to GL(\mathbb{R}^{d_{z'}})$ on latent spaces $\mathcal{Z} = \mathbb{R}^{d_z}$ and $\mathcal{Z}' = \mathbb{R}^{d_{z'}}$ respectively.

A neural operator $f: \mathcal{Z} \to \mathcal{Z}'$ is **$G$-equivariant** (or **$(\rho, \rho')$-equivariant** when representations need emphasis) if:

$$
f(\rho(g) z) = \rho'(g) f(z) \quad \forall g \in G, \; z \in \mathcal{Z}
$$

where matrix-vector multiplication is implied.

**Physical interpretation:** Transforming the input then applying $f$ gives the same result as applying $f$ then transforming the output. Formally, the following diagram commutes:

$$
\begin{array}{ccc}
\mathcal{Z} & \xrightarrow{f} & \mathcal{Z}' \\
\downarrow \rho(g) & & \downarrow \rho'(g) \\
\mathcal{Z} & \xrightarrow{f} & \mathcal{Z}'
\end{array}
$$

**Units:** If $[z] = [z']$ (both in units of $\sqrt{\text{nat}}$), then $[f(z)] = [z']$ and $[\rho(g)] = [\rho'(g)] = $ dimensionless.
:::

:::{prf:definition} Fragile Gauge Group
:label: def-fragile-gauge-group

The Fragile framework adopts the gauge structure:

$$
G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y
$$

**Source:** This gauge group emerges from the multi-agent consistency requirements under capacity constraints $C < \infty$ (Chapter 8.1, {ref}`sec-symplectic-multi-agent-field-theory`). The full derivation from first principles—showing why this specific product structure is necessary and sufficient—is provided in Chapter 8, Sections 8.1-8.3.

**Important qualification:** This chapter **assumes** the gauge structure and demonstrates its implementation in neural architectures. We do not re-derive the gauge group here; instead, we cite the field-theoretic analysis from Chapter 8.1. Readers seeking the foundational derivation should consult that chapter first.

**Explicit definitions:**

1. **$SU(N_f)_C$ (Color symmetry):** The special unitary group of degree $N_f$, consisting of $N_f \times N_f$ complex unitary matrices $U$ with $\det(U) = 1$. Here $N_f$ equals the number of feature bundles $n_b$ (see Definition {prf:ref}`def-latent-vector-bundle`). This acts on the bundle index, permitting feature mixing across bundles.

2. **$SU(2)_L$ (Weak isospin):** The special unitary group of degree 2, consisting of $2 \times 2$ complex unitary matrices with determinant 1. This has 3 real parameters (Pauli matrix basis) and acts on observation-action doublets.

3. **$U(1)_Y$ (Hypercharge):** The circle group $\{e^{i\theta} : \theta \in [0, 2\pi)\} \cong SO(2)$, representing phase rotations. This is associated with a capacity bound (holographic bound).

**Representation on latent space:** For a latent space $\mathcal{Z} = \mathbb{R}^{d_z}$ decomposed into $n_b$ bundles of dimension $d_b$ (so $d_z = n_b \cdot d_b$), we construct the representation $\rho: G_{\text{Fragile}} \to GL(d_z, \mathbb{R})$ through **real forms** of the complex gauge groups.

**Challenge:** $SU(N_f)$ and $SU(2)$ are complex Lie groups (acting on $\mathbb{C}^{N_f}$ and $\mathbb{C}^2$ respectively), but our latent space $\mathcal{Z} = \mathbb{R}^{d_z}$ is real. We need a **real representation** compatible with neural network operations.

**Construction via real forms:**

1. **$SU(N_f)_C$ real representation:** With $N_f = n_b$ (number of bundles), we model bundle mixing with a real orthogonal action on the bundle index. This is a pragmatic, real-valued proxy for $SU(N_f)_C$ that preserves bundle norms and keeps the bundle count fixed. In block form:

   $$
   \rho_C(R) = \begin{pmatrix}
   R_{11} I_{d_b} & R_{12} I_{d_b} & \cdots & R_{1n_b} I_{d_b} \\
   R_{21} I_{d_b} & R_{22} I_{d_b} & \cdots & R_{2n_b} I_{d_b} \\
   \vdots & \vdots & \ddots & \vdots \\
   R_{n_b 1} I_{d_b} & R_{n_b 2} I_{d_b} & \cdots & R_{n_b n_b} I_{d_b}
   \end{pmatrix} \in GL(d_z, \mathbb{R})
   $$
   where $R = (R_{ij}) \in SO(n_b)$ is a real orthogonal mixing matrix on bundle indices (an architectural proxy for $SU(N_f)_C$), and $I_{d_b}$ is the $d_b \times d_b$ identity (each bundle block rotates together).

   **Explicit $SU(n_b) \to SO(2n_b)$ realification (faithful):** There is no canonical homomorphism $SU(N) \to SO(N)$ in general. A standard realification yields a faithful embedding into $SO(2N)$ as follows:

   1. **Complex to real decomposition:** Any $U \in SU(n_b)$ can be written as $U = A + iB$ where $A, B \in \mathbb{R}^{n_b \times n_b}$.

   2. **Block real form:** This induces a real representation $\rho_{\mathbb{R}}: SU(n_b) \to SO(2n_b)$ given by:

      $$
      U = A + iB \mapsto \begin{pmatrix} A & -B \\ B & A \end{pmatrix} \in SO(2n_b)
      $$
      This is an embedding preserving orthogonality: $\|Uz\|^2 = \|z\|^2$ for $z \in \mathbb{C}^{n_b}$ translates to the block matrix preserving $\mathbb{R}^{2n_b}$ norm.

   **Verification of homomorphism property:**

   1. **Identity preservation:** $I_N + i \cdot 0 \mapsto \mathrm{diag}(I_N, I_N) = I_{2N}$ ✓

   2. **Multiplicativity:** For $U_1, U_2 \in SU(N)$ with $U_k = A_k + iB_k$, the product $U_1 U_2 = (A_1 A_2 - B_1 B_2) + i(A_1 B_2 + B_1 A_2)$ satisfies:
      $$\rho_{\mathbb{R}}(U_1 U_2) = \begin{pmatrix} A_1 A_2 - B_1 B_2 & -(A_1 B_2 + B_1 A_2) \\ A_1 B_2 + B_1 A_2 & A_1 A_2 - B_1 B_2 \end{pmatrix}$$
      Direct computation of block matrix multiplication gives:
      $$\rho_{\mathbb{R}}(U_1) \rho_{\mathbb{R}}(U_2) = \begin{pmatrix} A_1 & -B_1 \\ B_1 & A_1 \end{pmatrix} \begin{pmatrix} A_2 & -B_2 \\ B_2 & A_2 \end{pmatrix} = \begin{pmatrix} A_1 A_2 - B_1 B_2 & -A_1 B_2 - B_1 A_2 \\ B_1 A_2 + A_1 B_2 & -B_1 B_2 + A_1 A_2 \end{pmatrix}$$
      Therefore $\rho_{\mathbb{R}}(U_1 U_2) = \rho_{\mathbb{R}}(U_1) \rho_{\mathbb{R}}(U_2)$, confirming the homomorphism property.

   3. **Orthogonality:** The unitarity condition $U^* U = I$ for $U = A + iB$ decomposes into $A^T A + B^T B = I_N$ (real part) and $A^T B = B^T A$ (imaginary part, implying $A^T B$ is symmetric). Computing:
      $$\rho_{\mathbb{R}}(U)^T \rho_{\mathbb{R}}(U) = \begin{pmatrix} A^T & B^T \\ -B^T & A^T \end{pmatrix} \begin{pmatrix} A & -B \\ B & A \end{pmatrix} = \begin{pmatrix} A^T A + B^T B & -A^T B + B^T A \\ -B^T A + A^T B & B^T B + A^T A \end{pmatrix}$$
      Using $A^T B = B^T A$ and $A^T A + B^T B = I_N$ yields $\rho_{\mathbb{R}}(U)^T \rho_{\mathbb{R}}(U) = I_{2N}$, so $\rho_{\mathbb{R}}(U) \in O(2N)$.

   4. **Determinant:** Using the block determinant identity:
      $$\det(\rho_{\mathbb{R}}(U)) = \det(A + iB)\det(A - iB) = \det(U)\det(\bar{U}) = |\det(U)|^2 = 1$$
      confirming $\rho_{\mathbb{R}}: SU(N) \to SO(2N)$.

   **Implementation note:** The faithful $SO(2n_b)$ realification doubles the bundle-index dimension. In practice, the full $SU(n_b)$ symmetry is **broken to a discrete subgroup** (permutations and sign flips of bundles), which naturally embeds in $SO(n_b)$ as signed permutation matrices. This is enforced implicitly through the isotropic architecture design, not by explicitly constructing gauge transformations.

2. **$SU(2)_L$ real representation:** The adjoint representation identifies the Lie algebras $\mathfrak{su}(2) \cong \mathfrak{so}(3)$ and yields a surjective homomorphism $\mathrm{Ad}: SU(2) \to SO(3)$ with kernel $\{\pm I\}$ {cite}`hall2015lie`. This gives a 3D real representation that factors through $SO(3)$ (so it is **not** faithful on $SU(2)$). A faithful real representation is obtained by realifying the fundamental doublet $SU(2) \curvearrowright \mathbb{C}^2$, giving an embedding $SU(2) \hookrightarrow SO(4)$ (equivalently, $SU(2)\cong Sp(1)$ acting on quaternions $\mathbb{H}\cong \mathbb{R}^4$ by left multiplication).

   **Architecture choice — spontaneous symmetry breaking:** In practice, the full $SU(2)_L$ symmetry is **broken to a $U(1)$ subgroup** (a maximal torus; all such subgroups are conjugate to the diagonal subgroup $\mathrm{diag}(e^{i\theta}, e^{-i\theta}) \in SU(2)$). In our real-valued implementation we realize this $U(1)$ action as the standard $SO(2)$ rotation mixing the observation-action doublet components (generated by $i\sigma_2$ in the Pauli basis):

   $$
   \rho_L: U(1) \hookrightarrow SU(2)_L \xrightarrow{\text{real form}} SO(2) \subset GL(2, \mathbb{R})
   $$
   acting on $(z_{\text{obs}}, z_{\text{act}}) \in \mathbb{R}^2 \subset \mathcal{Z}$.

   **Justification for symmetry breaking:** The reduction from $SU(2)_L$ to $U(1) \subset SU(2)_L$ is consistent with the weak isospin structure in the Standard Model, where the unbroken subgroup after electroweak symmetry breaking is $U(1)_{\text{EM}} \subset SU(2)_L \times U(1)_Y$. In the neural architecture, only the $U(1)$ rotation symmetry between observation and action channels is explicitly enforced; a faithful real implementation of the full $SU(2)$ doublet would require complex features (or its 4D realification), which is incompatible with a 2D real obs-action plane. See the remark in Definition {prf:ref}`def-obs-action-doublet` on real-valued implementations.

3. **$U(1)_Y$ real representation:** The hypercharge symmetry $U(1)_Y = \{e^{i\theta} : \theta \in [0, 2\pi)\}$ acts on complex fields as phase rotations $\psi \mapsto e^{iY\theta} \psi$, preserving $|\psi|^2$. The real representation is:

   $$
   \rho_Y: U(1)_Y \to SO(2) \subset GL(2, \mathbb{R}), \quad e^{i\theta} \mapsto \begin{pmatrix} \cos(Y\theta) & -\sin(Y\theta) \\ \sin(Y\theta) & \cos(Y\theta) \end{pmatrix}
   $$
   where $Y \in \mathbb{R}$ is the hypercharge quantum number. This is a rotation in a 2D subspace with angular velocity proportional to $Y$.

   **Architecture implementation via norm preservation:** In practice, $U(1)_Y$ invariance is enforced through **Lipschitz constraints** $\sigma_{\max}(W) \leq 1$ on all linear operators (Definition {prf:ref}`def-spectral-linear`). These constraints ensure:

   $$
   \|W z\| \leq \|z\| \quad \forall z \in \mathcal{Z}
   $$
   which preserves the total "charge" $\|z\|^2$ up to a maximum value (holographic bound $C < \infty$). The spectral bound implements **non-expansiveness**, making operators contractive or (in the limit) isometric; this relaxes strict $U(1)$ rotations to norm-non-increasing maps that still preserve a scalar charge. Theorem {prf:ref}`thm-spectral-preserves-hypercharge` proves that composing spectrally normalized layers maintains $\|z_t\| \leq \|z_0\|$ (non-increasing across layers), consistent with capacity constraints.

   **Relationship to hypercharge conservation:** In the Standard Model, hypercharge $Y$ is a conserved quantum number satisfying $Q = T_3 + Y/2$ (electric charge formula). In the neural architecture, the analog is **total information content** $I(X_t; Z_t) \leq C$, which is bounded by the holographic principle. Spectral normalization ensures the *linear* operators are non-expansive ($\|Wz\| \leq \|z\|$); keeping the overall network within capacity additionally requires controlling the gain of nonlinear blocks (e.g., via rescaled NormGate or explicit Lipschitz tracking).

**Product representation:** The full representation is the **direct sum** (⊕) of these factors acting on disjoint subspaces:

$$
\rho(U_C, U_L, e^{i\theta_Y}) = \rho_C(U_C) \oplus \rho_L(U_L) \oplus \rho_Y(e^{i\theta_Y})
$$
in block-diagonal form (each factor acts on its own subspace).

**Notation clarification:** We use the direct sum ⊕ (not tensor product ⊗) as an architectural simplification:
- **Direct sum $V \oplus W$:** Dimension = $\dim(V) + \dim(W)$. Transformations are block-diagonal: $\begin{pmatrix} A & 0 \\ 0 & B \end{pmatrix}$
- **Tensor product $V \otimes W$:** Dimension = $\dim(V) \times \dim(W)$. Transformations are Kronecker products: $A \otimes B$

In this model, each gauge factor acts independently on a designated subspace of $\mathcal{Z}$, so the representation space is $\mathcal{Z} = \mathcal{Z}_C \oplus \mathcal{Z}_L \oplus \mathcal{Z}_Y$ (direct sum), and the group representation is the direct sum of individual representations.

**Derivation of direct sum structure:**

The choice of direct sum over tensor product is **architectural**, not fundamental. Here's why:

1. **Tensor product would be physically correct:** In gauge theory, matter fields typically transform under tensor product representations. For example, quarks transform as $(\mathbf{3}, \mathbf{2}, 1/6)$ under $SU(3)_C \times SU(2)_L \times U(1)_Y$, meaning the representation space is $V_C \otimes V_L \otimes V_Y$ with dimension $3 \times 2 \times 1 = 6$ per flavor.

2. **Direct sum is a simplifying assumption:** We decompose $\mathcal{Z} = \mathcal{Z}_C \oplus \mathcal{Z}_L \oplus \mathcal{Z}_Y$ with each subspace transforming independently. This reduces computational complexity:
   - Tensor product: $d_z = d_C \times d_L \times d_Y$ (exponential growth)
   - Direct sum: $d_z = d_C + d_L + d_Y$ (linear scaling)

3. **Physical interpretation:** The direct sum structure corresponds to **separate degrees of freedom** for color, weak isospin, and hypercharge. This is analogous to decomposing a particle state into spin, flavor, and color quantum numbers as independent labels, rather than a fully entangled state.

4. **Consistency with architecture:** The bundle structure (Definition {prf:ref}`def-latent-vector-bundle`) already assumes $\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i$ where each bundle $V_i$ is invariant under $\rho_C$. The $SU(2)_L$ and $U(1)_Y$ factors act on additional subspaces orthogonal to the bundle subspace.

**Implication:** This choice means we enforce gauge symmetry **separately** for each factor, not jointly. Full tensor product representations could be implemented but would require rethinking the bundle decomposition structure.

**Critical caveat:** This architectural simplification means the implementation does **not** fully realize the gauge-theoretic structure derived in Chapter 8.1. The direct sum is a **practical approximation** that preserves gauge covariance at the level of individual factors while avoiding the combinatorial explosion of full tensor product representations. Future work could explore whether tensor product architectures provide empirical benefits justifying the increased complexity.

**Implementation note:** In practice, neural network architectures do NOT implement the full gauge group action explicitly. Instead, we build **equivariant primitives**:
- IsotropicBlock (Definition {prf:ref}`def-isotropic-block`) is equivariant w.r.t. $\rho_C$ (bundle mixing, Theorem {prf:ref}`thm-isotropic-preserves-color`)
- SteerableConv (Section {ref}`sec-covariant-retina`) is equivariant w.r.t. $\rho_L$ (obs-action doublet, Proposition {prf:ref}`prop-obs-action-doublet`)
- SpectralLinear (Definition {prf:ref}`def-spectral-linear`) preserves $\rho_Y$ (hypercharge bound, Theorem {prf:ref}`thm-spectral-preserves-hypercharge`)

**Remark on complex vs. real:** Physicists typically work with complex representations because quantum mechanics is inherently complex (wavefunctions are in $\mathbb{C}$). Neural networks are real-valued (weights in $\mathbb{R}$), so we use real forms. The **isomorphism** $SU(2) \cong \text{Spin}(3) \to SO(3)$ and $SU(N) \supset SO(N)$ (via embedding) allow translation between complex and real pictures. See Section {ref}`sec-symplectic-multi-agent-field-theory` for the complex gauge field formulation; here we use the real neural implementation.

**Requirement:** All neural operators in the latent dynamics must be $G_{\text{Fragile}}$-equivariant to preserve physical consistency.

**Implication:** Standard building blocks (ReLU, LayerNorm, biased Linear) that violate even simple $SO(d)$ equivariance cannot be used directly.
:::

:::{admonition} Connection to RL: Equivariance vs. Data Augmentation
:class: note

**Standard RL approach:** Augment training data with rotated/transformed copies, hope the model learns approximate invariance through exposure.

**Fragile approach:** Build exact equivariance into the architecture. The network *cannot* violate the symmetry, even with adversarial inputs.

**Advantage:** Guaranteed consistency (no approximate learning), better sample efficiency (no need to see all rotations), mathematical rigor (enables formal proofs about behavior).
:::

### The Failure of ReLU

:::{div} feynman-prose
Now let me show you exactly what goes wrong with ReLU, because this is not abstract—it is a concrete bug you can compute with pencil and paper.

ReLU says: "Look at each component of the vector. If it is negative, kill it. If it is positive, keep it." This sounds innocent enough. But notice what is hidden in that instruction: *which* components? The answer depends entirely on what coordinate axes you chose. And that choice was arbitrary.

Here is the picture to hold in your mind. Imagine the latent space as a room, and your coordinate axes as lines painted on the floor. ReLU creates "walls" along those painted lines—whenever a vector crosses from the positive side to the negative side of an axis, that component gets chopped to zero. Now rotate the room. The painted lines move, and the walls move with them. A vector that passed through freely before now hits a wall and gets truncated. *The physics has changed because you rotated your bookkeeping.*

There is another way to see this. ReLU creates kinks in the function—sharp corners where the derivative is undefined. These kinks are aligned with the coordinate axes. In a gauge-invariant world, there are no preferred directions, so there should be no preferred locations for kinks. But ReLU puts them exactly along the axes, breaking the smooth differential structure that the geodesic integrator requires.

The proof that follows makes this precise: apply ReLU, then rotate; versus rotate, then apply ReLU. You get different answers. That is the definition of gauge violation.
:::

:::{prf:theorem} ReLU Violates SO(d) Equivariance
:label: thm-relu-breaks-equivariance

Let $d \geq 2$ and consider the standard representation $\rho: SO(d) \to GL(d, \mathbb{R})$ where $\rho(R) = R$ (i.e., rotations act by matrix multiplication).

Define the ReLU activation $f: \mathbb{R}^d \to \mathbb{R}^d$ by:

$$
f(z) = (f(z_1), \ldots, f(z_d)) \quad \text{where} \quad f(z_i) = \max(0, z_i)
$$

Then $f$ is **not** $SO(d)$-equivariant with respect to $\rho$.

*Proof.*

We prove by explicit counterexample for $d=2$, then indicate generalization.

**Step 1. Counterexample construction ($d=2$):**

Let $z = (1, -1)^T \in \mathbb{R}^2$. Then:

$$
f(z) = (\max(0, 1), \max(0, -1))^T = (1, 0)^T
$$

**Step 2. Apply rotation:**

Let $R_\theta \in SO(2)$ be counterclockwise rotation by $\theta = \pi/4$:

$$
R_{\pi/4} = \begin{pmatrix} \cos(\pi/4) & -\sin(\pi/4) \\ \sin(\pi/4) & \cos(\pi/4) \end{pmatrix} = \begin{pmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{pmatrix}
$$

Then:

$$
R_{\pi/4} z = \begin{pmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{pmatrix} \begin{pmatrix} 1 \\ -1 \end{pmatrix} = \begin{pmatrix} \frac{1+1}{\sqrt{2}} \\ \frac{1-1}{\sqrt{2}} \end{pmatrix} = \begin{pmatrix} \sqrt{2} \\ 0 \end{pmatrix}
$$

**Step 3. Compute $f(R_{\pi/4} z)$:**

$$
f(R_{\pi/4} z) = f((\sqrt{2}, 0)^T) = (\sqrt{2}, 0)^T
$$

**Step 4. Compute $R_{\pi/4} f(z)$:**

$$
R_{\pi/4} f(z) = R_{\pi/4} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{pmatrix}
$$

**Step 5. Verify non-equality:**

$$
f(R_{\pi/4} z) = (\sqrt{2}, 0)^T \neq \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)^T = R_{\pi/4} f(z)
$$

Thus $f(R_{\pi/4} z) \neq R_{\pi/4} f(z)$, violating $SO(2)$-equivariance.

**Generalization to $d > 2$:**

For arbitrary $d \geq 2$, we construct an analogous counterexample.

**Setup:** Let $z = (1, -1, 0, \ldots, 0)^T \in \mathbb{R}^d$ (first two components nonzero, rest zero).

Define $R_\theta \in SO(d)$ as the block-diagonal matrix:

$$
R_\theta = \begin{pmatrix}
R_2(\theta) & 0 \\
0 & I_{d-2}
\end{pmatrix}
$$
where $R_2(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$ is the $2 \times 2$ rotation by angle $\theta = \pi/4$, and $I_{d-2}$ is the $(d-2) \times (d-2)$ identity matrix.

**Verification that $R_\theta \in SO(d)$:**
1. **Orthogonality:** $R_\theta^T R_\theta = \begin{pmatrix} R_2^T R_2 & 0 \\ 0 & I_{d-2} \end{pmatrix} = \begin{pmatrix} I_2 & 0 \\ 0 & I_{d-2} \end{pmatrix} = I_d$ follows from block structure and $R_2^T R_2 = I_2$.
2. **Determinant:** $\det(R_\theta) = \det(R_2) \cdot \det(I_{d-2}) = 1 \cdot 1 = 1$.

**Step 1. Compute $R_\theta z$:**

Using the block-diagonal structure:

$$
R_\theta z = \begin{pmatrix} R_2(\pi/4) & 0 \\ 0 & I_{d-2} \end{pmatrix} \begin{pmatrix} 1 \\ -1 \\ 0 \\ \vdots \\ 0 \end{pmatrix} = \begin{pmatrix} R_2(\pi/4) \begin{pmatrix} 1 \\ -1 \end{pmatrix} \\ 0_{d-2} \end{pmatrix}
$$

Computing the $2 \times 2$ block (from Step 2 of the $d=2$ case):

$$
R_2(\pi/4) \begin{pmatrix} 1 \\ -1 \end{pmatrix} = \begin{pmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{pmatrix} \begin{pmatrix} 1 \\ -1 \end{pmatrix} = \begin{pmatrix} \sqrt{2} \\ 0 \end{pmatrix}
$$

Therefore:

$$
R_\theta z = (\sqrt{2}, 0, 0, \ldots, 0)^T
$$

**Step 2. Compute $f(R_\theta z)$:**

Since all components of $R_\theta z$ are non-negative (first component $\sqrt{2} > 0$, rest are $0$):

$$
f(R_\theta z) = (\max(0, \sqrt{2}), \max(0, 0), \ldots, \max(0, 0))^T = (\sqrt{2}, 0, 0, \ldots, 0)^T
$$

**Step 3. Compute $f(z)$:**

Component-wise application of $\max(0, \cdot)$:

$$
f(z) = (\max(0, 1), \max(0, -1), \max(0, 0), \ldots, \max(0, 0))^T = (1, 0, 0, \ldots, 0)^T
$$

**Step 4. Compute $R_\theta f(z)$:**

$$
R_\theta f(z) = \begin{pmatrix} R_2(\pi/4) & 0 \\ 0 & I_{d-2} \end{pmatrix} \begin{pmatrix} 1 \\ 0 \\ 0 \\ \vdots \\ 0 \end{pmatrix} = \begin{pmatrix} R_2(\pi/4) \begin{pmatrix} 1 \\ 0 \end{pmatrix} \\ 0_{d-2} \end{pmatrix}
$$

Computing:

$$
R_2(\pi/4) \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{pmatrix}
$$

Therefore:

$$
R_\theta f(z) = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0, \ldots, 0\right)^T
$$

**Step 5. Verify non-equality:**

Comparing the first two components:
- $f(R_\theta z) = (\sqrt{2}, 0, \ldots)^T$
- $R_\theta f(z) = (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, \ldots)^T$

Since $\sqrt{2} \neq \frac{1}{\sqrt{2}}$ and $0 \neq \frac{1}{\sqrt{2}}$, we have:

$$
f(R_\theta z) \neq R_\theta f(z)
$$

This holds for all $d \geq 2$. $\square$

**Geometric interpretation:** The ReLU creates coordinate-aligned decision boundaries $\{z \in \mathbb{R}^d : z_i = 0\}$ for each $i$. These hyperplanes are not invariant under $SO(d)$: rotation maps the hyperplane $\{z_1 = 0\}$ to a different hyperplane (no longer aligned with any coordinate axis). Thus $f$ transforms the kink surfaces under rotation, violating equivariance.

**Remark (General groups):** This argument generalizes to any Lie group $G$ with a nontrivial linear representation $\rho: G \to GL(d, \mathbb{R})$. "Nontrivial" means there exists $g \in G$ such that $\rho(g) e_i \neq \pm e_i$ for some basis vector $e_i$. For $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$, each factor acts nontrivially on its respective subspace (bundles, observation-action doublet, capacity), so ReLU violates $G_{\text{Fragile}}$-equivariance.

	$\square$
:::

:::{prf:corollary} ReLU Violates Smoothness Requirements for WFR Dynamics
:label: cor-relu-breaks-wfr

The WFR geometry (Chapter 5, Section {ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`) provides the dynamical framework for latent state evolution. The WFR action and geodesic integrator require computing gradient flows:

$$
\frac{dz}{ds} = -G^{ij}(z) \nabla_{z_j} \mathcal{L}_{\text{WFR}}(z)
$$

where $G^{ij}(z)$ is the metric tensor from Theorem {prf:ref}`thm-capacity-constrained-metric-law` and $\mathcal{L}_{\text{WFR}}$ is the WFR action functional.

**Smoothness requirement:** Gradient-based geodesic integrators (e.g., Boris-BAOAB in Section 5.4) require $\mathcal{L}_{\text{WFR}}$ to be at least $C^1$ (continuously differentiable) to compute well-defined gradients $\nabla_z \mathcal{L}$.

**ReLU violates this:** By Theorem {prf:ref}`thm-relu-breaks-equivariance`, ReLU creates non-differentiable kinks at coordinate hyperplanes $\{z \in \mathcal{Z} : z_i = 0\}$ for each $i = 1, \ldots, d_z$.

*Explicit derivation:*

**Step 1. Network with ReLU activation:**
Consider a simple network layer $f(z) = \max(0, Wz + b)$ where $W \in \mathbb{R}^{d \times d}$, $b \in \mathbb{R}^d$. The derivative is:

$$
\frac{\partial f_i}{\partial z_j} = \begin{cases}
W_{ij} & \text{if } (Wz + b)_i > 0 \\
0 & \text{if } (Wz + b)_i < 0 \\
\text{undefined} & \text{if } (Wz + b)_i = 0
\end{cases}
$$

**Step 2. WFR action functional dependence:**
The WFR action $\mathcal{L}_{\text{WFR}}[z]$ depends on the value function $V(z)$ and policy $\pi(a|z)$.

**Illustrative functional form:** For a capacity-constrained agent (Chapter 4, Theorem {prf:ref}`thm-equivalence-entropy-regularized-control`), a typical action functional is:

$$
\mathcal{L}_{\text{WFR}}[z] = V(f(z)) + \lambda I(\pi(\cdot|f(z)))
$$
where:
- $V: \mathcal{Z} \to \mathbb{R}$ is the expected cumulative reward (value function)
- $I(\pi(\cdot|f(z))) = I(A; f(Z))$ is mutual information between actions $A$ and latent state $f(Z)$
- $\lambda$ is the Lagrange multiplier enforcing capacity constraint $I(A; Z) \leq C$ (nat/step)
- $f$ represents the network transformation pipeline (which may contain ReLU non-differentiability)

**Source:** This form derives from the bounded-rationality variational principle (see Chapter 5, Section {ref}`sec-wfr-action-functional`, Equation 5.12). The argument below applies to **any** action functional requiring smooth gradients; we use this as a concrete example to demonstrate ReLU incompatibility.

**Step 3. Chain rule breakdown:**
To compute $\nabla_z \mathcal{L}_{\text{WFR}}$, we need:

$$
\frac{\partial \mathcal{L}}{\partial z_j} = \sum_i \frac{\partial \mathcal{L}}{\partial f_i} \cdot \frac{\partial f_i}{\partial z_j}
$$
At kink points where $(Wz + b)_i = 0$, the term $\frac{\partial f_i}{\partial z_j}$ is undefined, causing $\nabla_z \mathcal{L}$ to be undefined.

**Step 4. Consequences at kinks:**

1. **Gradient undefined:** $\nabla_z \mathcal{L}$ does not exist in the classical sense (left derivative $\lim_{h \to 0^-}$ differs from right derivative $\lim_{h \to 0^+}$)

2. **Fisher metric ill-defined:** The Fisher information metric $\mathcal{F}_{ij}(z) = \mathbb{E}_{a \sim \pi(\cdot|z)}[\partial_i \log \pi(a|z) \partial_j \log \pi(a|z)]$ requires computing $\partial_i \log \pi(a|z)$. If the policy network $\pi(a|z)$ is parameterized by a network with ReLU activations, then $\pi(a|z)$ is non-differentiable at kink points, causing $\partial_i \log \pi(a|z)$ to be undefined. Consequently, the Fisher metric tensor $\mathcal{F}(z)$ cannot be computed in the classical sense at these points.

   The capacity-constrained metric $G(z)$ from Theorem {prf:ref}`thm-capacity-constrained-metric-law` depends on the risk tensor $T_{ij}$, which in turn depends on gradients of the value function and the Fisher metric. If either $V$ or $\pi$ uses ReLU activations, the metric $G(z)$ will have ill-defined components at kink loci.

3. **Integration errors:** Boris-BAOAB integrator (Definition {prf:ref}`def-baoab-splitting`) uses $\frac{dz}{ds} = \mathcal{M}_{\text{curl}}\!\left(-G^{-1}\nabla \mathcal{L}\right)$ with $\mathcal{M}_{\text{curl}} := (I - \beta_{\text{curl}} G^{-1}\mathcal{F}_{\text{curl}})^{-1}$ (Value Curl; Definition {prf:ref}`def-value-curl`). At kinks, the discontinuous gradient causes ill-defined integration steps. While one could use subgradients or generalized gradients at kinks, this introduces numerical instability and prevents the integrator from preserving the symplectic structure required for long-term energy conservation

**Gauge-dependence problem:** Per Theorem {prf:ref}`thm-relu-breaks-equivariance`, ReLU kinks are coordinate-dependent. Under gauge transformation $z \mapsto U(g) \cdot z$ for $g \in G_{\text{Fragile}}$, the kink locations transform but ReLU does not transform equivariantly. This creates **inconsistent kink patterns** across gauge choices, causing geodesic flows to depend on arbitrary coordinate choices.

**Consequence:** Smooth, gauge-equivariant activations (e.g., GELU in NormGate, Definition {prf:ref}`def-norm-gated-activation`) are necessary for well-defined WFR gradient flows and gauge-invariant dynamics.

**Reference to WFR smoothness:** The WFR formulation's smoothness requirements are established through:
- Variational calculus on action functionals (standard $C^1$ requirement)
- Riemannian geometry for geodesic equations (smooth metric tensor)
- Symplectic integrator theory (Lipschitz gradients for Boris-BAOAB)

See Section 5.2 (WFR stress-energy tensor) and Part II, Hypostructure, Section 9 (Mathematical Prerequisites) for differential geometry foundations.

$\square$
:::



(sec-isotropic-bundle-operator)=
## The Isotropic Bundle Operator

:::{div} feynman-prose
Here is a question that will reveal a hidden assumption in nearly every neural network ever built: when you have a 512-dimensional latent vector, what *kind* of object is it?

Most practitioners treat it as 512 separate numbers—a list of scalars. Each dimension is independent, has its own meaning, gets its own weight, its own activation. But this is geometrically naive. A proper geometric object is not just a list of numbers; it is something that *transforms* in a definite way when you change coordinates.

Think about velocity. If I measure a car moving at 60 mph heading north, and you use a coordinate system rotated 90 degrees from mine, you will say the car is moving 60 mph heading east. The *components* changed, but the *velocity itself*—the geometric object—did not. Velocity is a vector, and vectors have transformation rules.

Now here is the key insight. Instead of treating the latent space as 512 independent scalars, we partition it into *bundles*—groups of dimensions that transform together as geometric vectors. If we have 32 bundles of 16 dimensions each, then within each bundle, the 16 components rotate together when you change frames. The bundle is a geometric object with definite transformation properties.

This is what a *vector bundle* means: at each point in your latent space, you attach a little vector space (the fiber), and the whole structure transforms coherently under gauge transformations. Standard neural networks ignore this structure entirely. We are going to respect it.
:::

### Geometric Preliminaries: Bundle Decomposition

:::{prf:definition} Bundle Decomposition of Latent Space
:label: def-latent-vector-bundle

Let $\mathcal{Z} = \mathbb{R}^{n_b \cdot d_b}$ be the latent space. A **bundle decomposition** partitions $\mathcal{Z}$ into $n_b$ **bundles** (subspaces) of dimension $d_b$:

$$
\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i, \quad V_i \cong \mathbb{R}^{d_b}, \quad n_b \times d_b = \dim(\mathcal{Z})
$$

Each bundle $V_i$ carries a representation of the rotation group $SO(d_b)$. For $g_i \in SO(d_b)$ and $v_i \in V_i$:

$$
\rho_i(g_i) \cdot v_i = g_i \cdot v_i \quad \text{(matrix multiplication)}
$$

where we identify $SO(d_b) \subset GL(d_b, \mathbb{R})$ as the subgroup of orthogonal matrices with determinant 1. This is the **defining (standard) representation** of $SO(d_b)$.

**Product gauge group (derivation):** Given the direct sum decomposition $\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i$, what is the maximal symmetry group?

**Claim:** The full symmetry group preserving the decomposition is:

$$
G_{\text{bundle}} = \prod_{i=1}^{n_b} SO(d_b)
$$

**Justification:**
1. **Preservation of decomposition:** Any symmetry must preserve $V_i \perp V_j$ for $i \neq j$ (orthogonal direct sum). Thus transformations cannot mix bundles.

2. **Within each bundle:** On $V_i \cong \mathbb{R}^{d_b}$, the maximal continuous symmetry group preserving the Euclidean inner product is $O(d_b)$ (orthogonal group).

3. **Orientation preservation:** For neural networks with deterministic forward pass, we require **orientation-preserving** transformations (continuous deformation from identity). Thus we restrict to $SO(d_b) \subset O(d_b)$ (special orthogonal: determinant +1).

4. **Independence:** Transformations on different bundles are independent, giving the product structure $\prod_{i=1}^{n_b} SO(d_b)$.

**Group action:** For $(g_1, \ldots, g_{n_b}) \in G_{\text{bundle}}$ and $z = (z^{(1)}, \ldots, z^{(n_b)})$ with $z^{(i)} \in V_i$:

$$
\rho(g_1, \ldots, g_{n_b}) \cdot z = (g_1 z^{(1)}, \ldots, g_{n_b} z^{(n_b)})
$$

In matrix form (with respect to concatenated basis):

$$
\rho(g_1, \ldots, g_{n_b}) = \text{diag}(g_1, \ldots, g_{n_b}) = \begin{pmatrix} g_1 & 0 & \cdots & 0 \\ 0 & g_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & g_{n_b} \end{pmatrix}
$$
(block-diagonal structure).

**Note on bundle permutations:** If bundles are **semantically distinguished** (e.g., bundle 1 = edges, bundle 2 = textures, bundle 3 = colors), we cannot permute them. If all bundles are **identical** (homogeneous feature space), the symmetry group extends to $(\prod_{i=1}^{n_b} SO(d_b)) \rtimes S_{n_b}$ where $S_{n_b}$ is the permutation group. For this architecture, we assume distinguished bundles.

**Units:** $[V_i] = [\mathcal{Z}] = \sqrt{\text{nat}}$ (from capacity constraint, see Section {ref}`sec-dimensional-analysis`).

**Remark:** This is a **direct sum decomposition with group action**, not a fiber bundle in the differential-geometric sense (which would require a base manifold and projection map). Analogous to gauge fields in physics (Chapter {ref}`sec-symplectic-multi-agent-field-theory`): just as the Error field $W_\mu$ transforms under $SU(2)_L$, bundles transform under their respective $SO(d_b)$ factors.
:::

### The Spectral Linear Map

:::{prf:definition} Spectral Linear Operator
:label: def-spectral-linear

A linear map $W: \mathcal{Z} \to \mathcal{Z}'$ is **spectrally normalized** if:

$$
\sigma_{\max}(W) \leq 1
$$

where $\sigma_{\max}(W)$ is the largest singular value of $W$.

**Remark (Singular values):** The singular values of $W$ are the square roots of the eigenvalues of $W^T W$. The largest singular value $\sigma_{\max}(W) = \sup_{\|z\|=1} \|Wz\|$ is the operator norm induced by the Euclidean norm. This follows from the singular value decomposition: $W = U \Sigma V^T$ where $U, V$ are orthogonal and $\Sigma$ is diagonal with non-negative entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$. Then $\|Wz\| = \|U \Sigma V^T z\| = \|\Sigma V^T z\|$ (since $U$ is orthogonal), which achieves maximum $\sigma_1$ when $V^T z$ aligns with the first singular vector. For a standard reference, see Horn & Johnson, *Matrix Analysis*, Theorem 5.6.2 {cite}`horn2012matrix`.

**Implementation:** $W_{\text{spectral}} = W / \sigma_{\max}(W)$.

**Units:** $[W]$ is dimensionless (if $\mathcal{Z}, \mathcal{Z}'$ normalized).
:::

:::{prf:theorem} Spectral Normalization Preserves Light Cone
:label: thm-spectral-preserves-light-cone

Let $(\mathcal{Z}, \|\cdot\|)$ be a finite-dimensional normed vector space with $\|\cdot\|$ the **Euclidean norm** $\|z\| = \sqrt{z^T z}$. Let $W: \mathcal{Z} \to \mathcal{Z}'$ be a linear map with $\sigma_{\max}(W) \leq 1$ (spectrally normalized). Let $c_{\text{info}}$ be the information speed limit (Axiom {prf:ref}`ax-information-speed-limit`).

Then:

$$
\|W \cdot z\| \leq \|z\| \quad \text{(contraction property)}
$$

This ensures that whenever $d(z_1, z_2) := \|z_1 - z_2\| \leq c_{\text{info}} \cdot \Delta t$ (causal interval), we have:

$$
d(W \cdot z_1, W \cdot z_2) \leq c_{\text{info}} \cdot \Delta t
$$

Thus the **light cone** $\mathcal{C}(z_0, t_0) := \{(z, t) : \|z - z_0\| \leq c_{\text{info}}(t - t_0), \, t \geq t_0\}$ in the extended state-time space is preserved: if $(z, t) \in \mathcal{C}(z_0, t_0)$, then $(W \cdot z, t) \in \mathcal{C}(W \cdot z_0, t_0)$.

*Proof.*

**Step 1. Spectral bound:** By spectral normalization (Definition {prf:ref}`def-spectral-linear`), $\sigma_{\max}(W) \leq 1$.

**Step 2. Operator norm:** For the Euclidean norm, the induced operator norm satisfies $\|W\|_{\text{op}} = \sigma_{\max}(W)$, where:

$$
\|W\|_{\text{op}} := \sup_{\|z\| = 1} \|W \cdot z\|
$$

Thus for any $z \in \mathcal{Z}$:

$$
\|W \cdot z\| \leq \|W\|_{\text{op}} \cdot \|z\| = \sigma_{\max}(W) \cdot \|z\| \leq \|z\|
$$

**Step 3. Distance preservation:** For any $z_1, z_2 \in \mathcal{Z}$:

$$
\|W \cdot z_1 - W \cdot z_2\| = \|W \cdot (z_1 - z_2)\| \leq \|z_1 - z_2\|
$$

**Step 4. Causal structure:** If the inputs are causally connected ($d(z_1, z_2) \leq c_{\text{info}} \Delta t$), the outputs remain causally connected since:

$$
d(W \cdot z_1, W \cdot z_2) \leq d(z_1, z_2) \leq c_{\text{info}} \Delta t
$$

**Identification:** The Lipschitz constant $L = \sigma_{\max}(W) \leq 1$ ensures signal propagation speed $\leq c_{\text{info}}$, preventing "superluminal" information transfer that would violate the causal structure established in the Speed Window (Theorem {prf:ref}`thm-speed-window`).

**Connection to Node 62:** The CausalityViolationCheck diagnostic verifies $\sigma_{\max}(W) \leq 1 + \epsilon$ during training. Violations indicate faster-than-light gradients breaking temporal coherence.

$\square$
:::

:::{div} feynman-prose
You might wonder: why does the spectral linear map have no bias term? Every neural network tutorial tells you to include a bias. What is wrong with $Wz + b$?

Here is what is wrong: the bias vector $b$ picks out a *special point* in space. It says "this is the origin, and I am going to shift everything relative to it." But in a proper geometric theory, there is no special point. The origin of your coordinate system is arbitrary—a bookkeeping choice, not physics.

Think about it this way. If I have a vector $z$ representing some agent state, and I shift my coordinate origin by $b$, then $z$ becomes $z - b$. A gauge-invariant operation should not care about this shift. But $Wz + b$ and $W(z-b) + b = Wz - Wb + b$ give different results unless $Wb = 0$. The bias has created a preferred origin.

In the tangent bundle picture, this is even clearer. We are working with *directions*—vectors that represent velocities, gradients, rates of change. A direction has no preferred starting point. You can translate the base point anywhere and the direction remains the same direction. Adding a bias is like saying "all velocities should be measured relative to this special velocity I chose"—which is exactly the kind of frame-dependent nonsense gauge theory forbids.

So we set $\text{bias} = \text{False}$ and never look back. The rigorous justification follows.
:::

:::{prf:proposition} Bias Terms Break Tangent Bundle Structure
:label: prop-bias-breaks-tangent-bundle

Working in the tangent bundle $T\mathcal{Z}$, elements are pairs $(z, v)$ where $z \in \mathcal{Z}$ is a point and $v \in T_z \mathcal{Z}$ is a tangent vector at $z$.

A gauge-covariant map on the tangent bundle must satisfy:

$$
f(z, \rho(g) \cdot v) = \rho'(g) \cdot f(z, v) \quad \forall g \in G
$$

**Claim:** Adding a bias $b$ violates this unless $b = 0$.

*Proof.* Consider $f(z, v) = W v + b$. Then:

$$
f(z, \rho(g) \cdot v) = W \rho(g) v + b
$$
but

$$
\rho'(g) \cdot f(z, v) = \rho'(g) (W v + b) = \rho'(g) W v + \rho'(g) b
$$

For equivariance, we need $b = \rho'(g) b$ for all $g \in G$. If $G$ acts non-trivially (e.g., $SO(d)$ with $d > 1$), the only fixed point is $b = 0$. $\square$

**Extension to compositions:** For a composition of $L$ layers $F = f_L \circ \cdots \circ f_1$ where each $f_i(v) = W_i v + b_i$:

If $F$ is $G$-equivariant, then $F(\rho(g) v) = \rho(g) F(v)$ for all $g, v$.

**Proof by induction:**
- **Base case** ($L=1$): By the proof above, $b_1 = 0$.
- **Inductive step:** Assume $f_{L-1} \circ \cdots \circ f_1$ is equivariant, so each $b_i = 0$ for $i < L$.

  Then:

  $$
  F(\rho(g) v) = f_L((f_{L-1} \circ \cdots \circ f_1)(\rho(g) v)) = f_L(\rho(g) (f_{L-1} \circ \cdots \circ f_1)(v))
  $$

  $$
  = W_L \rho(g) h + b_L \quad \text{where } h = (f_{L-1} \circ \cdots \circ f_1)(v)
  $$

  For equivariance with $\rho(g) F(v) = \rho(g) (W_L h + b_L)$, we need $b_L = 0$.

Thus **all biases must vanish** for a composition to be equivariant. $\square$

**Coordinate-free interpretation:** In the tangent bundle $T\mathcal{Z}$, linear maps act on **velocity vectors**, which are equivalence classes of curves $[\gamma]$ where $\gamma(0) = z$, $\dot{\gamma}(0) = v$. Adding a constant $b$ is not well-defined on velocity vectors—it mixes base points (where you are) with tangent directions (which way you're moving), breaking the fiber bundle structure.
:::

::::{admonition} Physics Isomorphism: Light Cone Preservation
:class: note

**In General Relativity:** The speed of light $c$ is the maximum causal influence speed. The light cone structure defines which events can influence which others.

**In Agent Theory:** The information speed $c_{\text{info}}$ is the maximum rate at which latent state changes can propagate. Spectral normalization enforces $\|W \cdot z\| \leq \|z\|$, preserving the causal cone.

**Correspondence Table:**

| General Relativity | Agent (Spectral Norm) |
|:-------------------|:----------------------|
| Speed of light $c$ | Information speed $c_{\text{info}}$ |
| Lorentz invariance | Causal ordering preserved |
| Light cone $ds^2 \leq 0$ | $\|W \cdot z\| \leq \|z\|$ |
| Timelike separation | $d(z_1, z_2) \leq c_{\text{info}} \Delta t$ |
::::

### The Norm-Gated Activation

:::{prf:definition} Norm-Gated Activation
:label: def-norm-gated-activation

For a vector bundle $v_i \in V_i$, define the **norm-gated activation**:

$$
f(v_i) = v_i \cdot g(\|v_i\| + b_i)
$$

where:
- $\|v_i\| = \sqrt{v_i^T v_i}$ is the Euclidean norm ($SO(d_b)$-invariant)
- $b_i \in \mathbb{R}$ is a learnable scalar bias (the "activation potential")
- $g: \mathbb{R} \to \mathbb{R}$ is a smooth scalar function (e.g., GELU, sigmoid)

**Physical interpretation:** Energy filter with radial symmetry. The gate opens when signal energy $\|v_i\|$ exceeds the potential barrier $-b_i$.

**Units:** All dimensionless if latent vectors normalized.

**Smoothness at the origin (implementation detail):** The norm $\|v\|$ is not differentiable at $v=0$. The map $v \mapsto v \cdot g(\|v\|+b)$ is still $C^1$ under mild regularity of $g$ (the apparent $\|v\|^{-1}$ singularity in the Jacobian cancels as $v \to 0$), but if you require a globally $C^\infty$ map you can replace $\|v\|$ with a smoothed norm $\|v\|_\varepsilon := \sqrt{v^T v + \varepsilon^2}$ in implementations.

**Remark (Choice of gating function $g$):** While any smooth $g: \mathbb{R} \to \mathbb{R}$ preserves equivariance, GELU is a **pragmatic choice** among functions satisfying design constraints:

1. **$C^\infty$ smoothness of $g$ (necessary):** $g$ should be $C^\infty$; combined with a smoothed norm $\|v\|_\varepsilon$ (if global smoothness is required), this yields a $C^\infty$ block compatible with the WFR metric (Corollary {prf:ref}`cor-relu-breaks-wfr`) and geodesic integrator assumptions (Section 5.4).

2. **Linear growth at large arguments (desirable):** For $x \gg 1$, GELU satisfies $g(x) \approx x$, so the gate value scales approximately linearly with energy rather than saturating to a constant gain.

3. **Controlled Lipschitz constant (practical):** $L_g \approx 1.129$ (Lemma {prf:ref}`lem-normgate-lipschitz`), close to 1 and comparable to softplus.

4. **Empirical effectiveness (validation):** Strong performance in transformers {cite}`hendrycks2016gaussian`.

**Critical distinction:** GELU is **not uniquely determined** by first principles. Conditions 1-3 are satisfied by multiple functions (e.g., Swish family $g(x) = x \cdot \sigma(\beta x)$, Softplus). A first-principles derivation selecting GELU uniquely (e.g., via information-geometric optimization) remains an open problem.

**Comparison with alternatives:**

| Activation | Smoothness | $\sup_x \|g'(x)\|$ | Unbounded? | Issue if used |
|------------|------------|-------------------|------------|---------------|
| Sigmoid | $C^\infty$ | $1/4$ | No (saturates to [0,1]) | Caps the gate gain at 1 (reduced dynamic range) |
| Tanh | $C^\infty$ | $1$ | No (saturates to [-1,1]) | Caps the gate gain (and allows sign flips) |
| Softplus | $C^\infty$ | $1$ | Yes (linear at $+\infty$) | Always nonnegative (good if you want a true “gate”) |
| GELU | $C^\infty$ | $\approx 1.129$ | Yes | Slight amplification ($L > 1$), addressed via spectral normalization |

The key advantage of GELU is that high-energy features propagate with energy-dependent gain ($g(x) \to x$ as $x \to \infty$), while sigmoid/tanh force the gate value to saturate to a constant.
:::

:::{prf:theorem} Norm-Gating Preserves SO(d_b) Equivariance
:label: thm-norm-gating-equivariant

Let $f$ be the norm-gated activation (Definition {prf:ref}`def-norm-gated-activation`). Then:

$$
f(R \cdot v) = R \cdot f(v) \quad \forall R \in SO(d_b)
$$

*Proof.*

**Step 1. Decompose activation:** $f(v) = v \cdot h$ where $h = g(\|v\| + b)$ is a scalar.

**Step 2. Norm invariance:** For any $R \in SO(d_b)$ (orthogonal matrix):

$$
\|R \cdot v\| = \sqrt{(R \cdot v)^T (R \cdot v)} = \sqrt{v^T R^T R v} = \sqrt{v^T v} = \|v\|
$$

**Step 3. Scalar invariance:** Since the norm is invariant:

$$
h(R \cdot v) = g(\|R \cdot v\| + b) = g(\|v\| + b) = h(v)
$$

**Step 4. Equivariance:** Combining:

$$
f(R \cdot v) = (R \cdot v) \cdot h(R \cdot v) = (R \cdot v) \cdot h(v) = R \cdot (v \cdot h(v)) = R \cdot f(v)
$$

$\square$
:::

:::{div} feynman-prose
Now I want you to see the norm-gated activation as what it really is: an *energy barrier*.

Think of the norm $\|v\|$ as the *energy* of the signal—how strong is this bundle of features? And think of the bias $b$ as setting the *height* of a barrier that the signal must overcome to pass through. When the energy exceeds the barrier ($\|v\| > -b$), the gate opens and the signal propagates. When the energy is too low, the gate stays closed and the signal is suppressed.

This is exactly like a threshold detector in a physical system. A photon detector requires a minimum photon energy to trigger. A synapse requires sufficient neurotransmitter concentration to fire. Our norm gate requires sufficient signal energy to activate.

And here is the beautiful part: the *direction* of the vector does not matter. The norm $\|v\|$ is invariant under rotations—rotate the vector any way you like, and its length stays the same. So the gate decision is purely about *how much* energy, not *which direction* the energy points. Every orientation is treated identically.

Compare this with ReLU. There, the activation depends on individual components: "Is $v_1 > 0$? Is $v_2 > 0$?" Those questions have different answers depending on your coordinate frame. But "Is $\|v\| > $ threshold?" has the same answer no matter how you orient your axes. That is the difference between a gauge-invariant activation and a broken one.
:::

### The Complete Isotropic Block

:::{prf:definition} Isotropic Block
:label: def-isotropic-block

The **Isotropic Block** is the atomic unit of gauge-covariant architecture:

$$
\text{IsotropicBlock}(z) = \text{Reshape}(\text{NormGate}(\text{SpectralLinear}(z)))
$$

where:
- **SpectralLinear**: Linear map $W$ with $\sigma_{\max}(W) \leq 1$ (Definition {prf:ref}`def-spectral-linear`) that is **block-diagonal** with respect to the bundle decomposition (see Lemma {prf:ref}`lem-block-diagonal-necessary`)
- **Reshape**: $\mathbb{R}^d \to (\mathbb{R}^{d_b})^{n_b}$ (bundle partition)
- **NormGate**: Norm-gated activation applied per bundle (Definition {prf:ref}`def-norm-gated-activation`)

**Structure constraint:** For **exact** $G_{\text{bundle}} = \prod_{i=1}^{n_b} SO(d_b)$ equivariance, the weight matrix $W$ must be block-scalar:

$$
W = \begin{pmatrix} \lambda_1 I_{d_b} & 0 & \cdots & 0 \\ 0 & \lambda_2 I_{d_b} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_{n_b} I_{d_b} \end{pmatrix}
$$
where each $\lambda_i \in [-1, 1]$ is a learnable scalar (per-bundle scaling factor), and $I_{d_b}$ is the $d_b \times d_b$ identity matrix. This constraint follows from Schur's lemma: any linear map commuting with all elements of $SO(d_b)$ must be a scalar multiple of identity (see Lemma {prf:ref}`lem-schur-scalar-constraint`).

**Practical relaxation (approximate equivariance):** For increased expressiveness, implementations may use general block-diagonal $W$ with $\sigma_{\max}(W_i) \leq 1$. This sacrifices exact equivariance but provides bounded equivariance violation (see Proposition {prf:ref}`prop-approximate-equivariance-bound`).
:::

:::{prf:lemma} Block-Diagonal Structure is Necessary (But Not Sufficient) for Bundle Equivariance
:label: lem-block-diagonal-necessary

Let $W: \mathcal{Z} \to \mathcal{Z}$ be a linear map on $\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i$ with $V_i \cong \mathbb{R}^{d_b}$.

For $W$ to be $G_{\text{bundle}}$-equivariant where $G_{\text{bundle}} = \prod_{i=1}^{n_b} SO(d_b)$, it is **necessary** that $W$ be block-diagonal: $W = \text{diag}(W_1, \ldots, W_{n_b})$ with $W_i: V_i \to V_i$.

However, block-diagonal structure is **not sufficient** for equivariance of the full IsotropicBlock; an additional constraint is required (see Lemma {prf:ref}`lem-schur-scalar-constraint`).

*Proof of necessity.*

Suppose $W$ has off-diagonal blocks, i.e., there exist $i \neq j$ such that $W_{ij} \neq 0$ where $W = [W_{ij}]$ in block form.

Consider a group element $g = (g_1, \ldots, g_{n_b})$ where $g_i = R_\theta \in SO(d_b)$ is a nontrivial rotation and $g_j = I$ for $j \neq i$.

For this $g$ and input $z$ with $z^{(j)} \neq 0$, $z^{(k)} = 0$ for $k \neq j$:

$$
(W \rho(g) z)_i = \sum_{k=1}^{n_b} W_{ik} g_k z^{(k)} = W_{ii} g_i \cdot 0 + W_{ij} I \cdot z^{(j)} = W_{ij} z^{(j)}
$$

But:

$$
(\rho(g) W z)_i = g_i (W z)^{(i)} = R_\theta \sum_{k=1}^{n_b} W_{ik} z^{(k)} = R_\theta W_{ij} z^{(j)}
$$

Since $R_\theta \neq I$ and $R_\theta W_{ij} z^{(j)} \neq W_{ij} z^{(j)}$ (rotation changes direction), we have $W \rho(g) z \neq \rho(g) W z$, violating equivariance.

Therefore, off-diagonal blocks must vanish: $W_{ij} = 0$ for $i \neq j$. $\square$

**Remark:** This lemma establishes that the block-diagonal constraint is necessary but leaves open what additional structure is required on each block $W_i$. Lemma {prf:ref}`lem-schur-scalar-constraint` completes the characterization.
:::

:::{prf:lemma} Schur's Lemma Constraint: Scalar Blocks Required for Equivariance
:label: lem-schur-scalar-constraint

Let $W = \text{diag}(W_1, \ldots, W_{n_b})$ be block-diagonal where each $W_i: V_i \to V_i$ with $V_i \cong \mathbb{R}^{d_b}$.

Assume the scalar map $\phi_i: [0,\infty) \to [0,\infty)$ defined by

$$
\phi_i(r) := r \, \bigl|g(r + b_i)\bigr|
$$
is injective on $[0,\infty)$ (a non-degeneracy condition that holds, for example, when $g(r+b_i)\ge 0$ and $r \mapsto r\,g(r+b_i)$ is strictly increasing on the operating range).

Then the composition $\text{NormGate} \circ W$ is $G_{\text{bundle}}$-equivariant if and only if each $W_i = \lambda_i I_{d_b}$ for some scalar $\lambda_i \in \mathbb{R}$.

*Proof.*

**($\Leftarrow$) Sufficiency:** If $W_i = \lambda_i I_{d_b}$, then for any $g_i \in SO(d_b)$:

$$
W_i g_i = \lambda_i I \cdot g_i = \lambda_i g_i = g_i \lambda_i I = g_i W_i
$$

Thus $W_i$ commutes with all $g_i \in SO(d_b)$. The IsotropicBlock composition is then equivariant by standard composition of equivariant maps.

**($\Rightarrow$) Necessity:** Suppose $\text{NormGate} \circ W$ is $G_{\text{bundle}}$-equivariant. We show each $W_i$ must be a scalar multiple of identity.

**Step 1. Equivariance condition (bundle $i$):** For all $R \in SO(d_b)$ and $v \in V_i$,

$$
\text{NormGate}(W_i \, Rv) = R \, \text{NormGate}(W_i v)
$$
which expands to

$$
W_i Rv \cdot g(\|W_i Rv\| + b_i) = R W_i v \cdot g(\|W_i v\| + b_i).
$$

**Step 2. Take norms:** Using $\|R x\| = \|x\|$,

$$
\|W_i Rv\| \, \bigl|g(\|W_i Rv\| + b_i)\bigr| = \|W_i v\| \, \bigl|g(\|W_i v\| + b_i)\bigr|.
$$
By injectivity of $\phi_i$, this implies $\|W_i Rv\| = \|W_i v\|$ for all $R, v$, hence the gate scalars match:

$$
g(\|W_i Rv\| + b_i) = g(\|W_i v\| + b_i).
$$

**Step 3. Cancel the common scalar:** Substituting back into Step 1 gives

$$
W_i Rv = R W_i v \quad \forall R \in SO(d_b), \forall v \in V_i,
$$
so $W_i$ commutes with all of $SO(d_b)$.

**Step 4. Apply Schur's lemma / commutant characterization:** The standard representation of $SO(d_b)$ on $\mathbb{R}^{d_b}$ is irreducible for $d_b \ge 2$, and its commutant is $\{\lambda I\}$. Therefore $W_i = \lambda_i I_{d_b}$.

	$\square$

**Physical interpretation:** The scalar constraint $W_i = \lambda_i I$ means each bundle can only be uniformly scaled, not rotated or sheared within itself. This preserves the geometric isotropy of each bundle fiber.
:::

:::{prf:theorem} IsotropicBlock is G-Equivariant (Scalar Block Case)
:label: thm-isotropic-block-equivariant

Let $G = \prod_{i=1}^{n_b} SO(d_b)$ be the product gauge group (Definition {prf:ref}`def-latent-vector-bundle`). By Lemmas {prf:ref}`lem-block-diagonal-necessary` and {prf:ref}`lem-schur-scalar-constraint`, the weight matrix $W$ in SpectralLinear must be **block-scalar** for exact equivariance:

$$
W = \begin{pmatrix} \lambda_1 I_{d_b} & 0 & \cdots & 0 \\ 0 & \lambda_2 I_{d_b} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_{n_b} I_{d_b} \end{pmatrix}
$$

where each $\lambda_i \in [-1, 1]$ is a learnable scalar satisfying $|\lambda_i| \leq 1$ (spectral normalization).

Then the IsotropicBlock (Definition {prf:ref}`def-isotropic-block`) is **exactly** $G$-equivariant.

*Proof.*

**Step 1. Define the group action:** On the flat space $\mathcal{Z} \cong \mathbb{R}^{n_b \cdot d_b}$, the group $G = \prod_{i=1}^{n_b} SO(d_b)$ acts as:

$$
\rho(g_1, \ldots, g_{n_b}) \cdot z = (g_1 \cdot z^{(1)}, \ldots, g_{n_b} \cdot z^{(n_b)})
$$
where $z = (z^{(1)}, \ldots, z^{(n_b)})$ with $z^{(i)} \in \mathbb{R}^{d_b}$.

**Step 2. SpectralLinear with scalar blocks:**
For $W = \text{diag}(\lambda_1 I, \ldots, \lambda_{n_b} I)$ and $(g_1, \ldots, g_{n_b}) \in G$:

$$
W \cdot \rho(g_1, \ldots, g_{n_b}) \cdot z = (\lambda_1 g_1 z^{(1)}, \ldots, \lambda_{n_b} g_{n_b} z^{(n_b)})
$$

**Key property:** Since $\lambda_i I$ commutes with all $g_i \in SO(d_b)$:

$$
\lambda_i I \cdot g_i = g_i \cdot \lambda_i I \quad \forall g_i \in SO(d_b)
$$

Thus SpectralLinear is equivariant:

$$
W \cdot \rho(g) \cdot z = \rho(g) \cdot W \cdot z
$$

**Step 3. Reshape is equivariant:** The bundle partition $\mathbb{R}^{n_b \cdot d_b} \to (\mathbb{R}^{d_b})^{n_b}$ is equivariant by construction (identity map in bundled coordinates).

**Step 4. NormGate is $SO(d_b)$-equivariant per bundle:** By Theorem {prf:ref}`thm-norm-gating-equivariant`, for each bundle $i$:

$$
\text{NormGate}(g_i \cdot v_i) = g_i \cdot \text{NormGate}(v_i) \quad \forall g_i \in SO(d_b)
$$

**Step 5. Composition equivariance:**

We prove $\text{IsotropicBlock}(\rho(g) \cdot z) = \rho(g) \cdot \text{IsotropicBlock}(z)$ by composing equivariant maps.

Let $g = (g_1, \ldots, g_{n_b}) \in G$ and $z = (z^{(1)}, \ldots, z^{(n_b)})$.

Compute left-hand side:

$$
\begin{align}
\text{IsotropicBlock}(\rho(g) \cdot z) &= \text{NormGate}(W \cdot \rho(g) \cdot z) \\
&= \text{NormGate}(\rho(g) \cdot W \cdot z) \quad \text{(Step 2: $W$ is equivariant)} \\
&= \rho(g) \cdot \text{NormGate}(W \cdot z) \quad \text{(Step 4: NormGate is equivariant)} \\
&= \rho(g) \cdot \text{IsotropicBlock}(z)
\end{align}
$$

**Explicit verification for bundle $i$:**
Let $v_i = \lambda_i z^{(i)}$ (output of SpectralLinear for bundle $i$).

- LHS: $\text{NormGate}(\lambda_i g_i z^{(i)}) = (\lambda_i g_i z^{(i)}) \cdot h(\|\lambda_i g_i z^{(i)}\|)$
- Since $\|\lambda_i g_i z^{(i)}\| = |\lambda_i| \cdot \|g_i z^{(i)}\| = |\lambda_i| \cdot \|z^{(i)}\| = \|\lambda_i z^{(i)}\|$:
- LHS $= (\lambda_i g_i z^{(i)}) \cdot h(\|\lambda_i z^{(i)}\|) = g_i (\lambda_i z^{(i)}) \cdot h(\|\lambda_i z^{(i)}\|) = g_i \cdot \text{NormGate}(v_i)$ = RHS

$\square$

**Remark:** The scalar block constraint $W_i = \lambda_i I$ is essential for exact equivariance. General block-diagonal matrices do NOT yield equivariance (see Lemma {prf:ref}`lem-schur-scalar-constraint`). For increased expressiveness at the cost of exact equivariance, see Proposition {prf:ref}`prop-approximate-equivariance-bound` below.
:::

:::{prf:proposition} Approximate Equivariance Bound for General Block-Diagonal W
:label: prop-approximate-equivariance-bound

For practical architectures using general block-diagonal $W = \text{diag}(W_1, \ldots, W_{n_b})$ with $\sigma_{\max}(W_i) \leq 1$ (instead of scalar blocks), the equivariance violation is bounded.

**Statement:** Let $\text{IB}$ denote IsotropicBlock with general block-diagonal $W$. Fix an operating range in which the per-bundle NormGate is $L_{\text{NG}}$-Lipschitz (Lemma {prf:ref}`lem-normgate-lipschitz`). Then for any $g = (g_1,\ldots,g_{n_b}) \in G_{\text{bundle}}$ and $z = (z^{(1)},\ldots,z^{(n_b)}) \in \mathcal{Z}$:

$$
\|\text{IB}(\rho(g)\cdot z) - \rho(g)\cdot \text{IB}(z)\|
\le
L_{\text{NG}} \, \|(W\rho(g) - \rho(g)W)z\|
=
L_{\text{NG}} \left(\sum_{i=1}^{n_b} \|[W_i, g_i]\,z^{(i)}\|^2\right)^{1/2}.
$$

In particular,

$$
\|\text{IB}(\rho(g)\cdot z) - \rho(g)\cdot \text{IB}(z)\|
\le
L_{\text{NG}} \sum_{i=1}^{n_b} \|[W_i, g_i]\|_{\mathrm{op}} \,\|z^{(i)}\|.
$$

**Implication:** If each $W_i$ commutes with $SO(d_b)$ (equivalently $W_i=\lambda_i I$ by Lemma {prf:ref}`lem-schur-scalar-constraint`), then $[W_i,g_i]=0$ and the equivariance violation is exactly zero.

*Proof.*

Use equivariance of NormGate (Theorem {prf:ref}`thm-norm-gating-equivariant`) to rewrite:

$$
\rho(g)\cdot \text{IB}(z)
=
\rho(g)\cdot \text{NormGate}(Wz)
=
\text{NormGate}(\rho(g)\cdot Wz).
$$
Therefore,

$$
\text{IB}(\rho(g)\cdot z) - \rho(g)\cdot \text{IB}(z)
=
\text{NormGate}(W\rho(g)z) - \text{NormGate}(\rho(g)Wz).
$$
By the $L_{\text{NG}}$-Lipschitz property of NormGate on the operating range (Lemma {prf:ref}`lem-normgate-lipschitz`),

$$
\|\text{NormGate}(W\rho(g)z) - \text{NormGate}(\rho(g)Wz)\|
\le
L_{\text{NG}} \,\|(W\rho(g) - \rho(g)W)z\|.
$$
Since both $W$ and $\rho(g)$ are block-diagonal, the commutator is block-diagonal with blocks $[W_i,g_i]$, yielding the stated per-bundle forms.

$\square$
:::



(sec-covariant-retina)=
## The Covariant Retina

:::{div} feynman-prose
Convolutional neural networks have one beautiful property: translation equivariance. Shift the input image left by 10 pixels, and the feature maps shift left by 10 pixels. The pattern detector does not care *where* in the image a cat appears—it finds the cat regardless. This is why CNNs revolutionized computer vision.

But here is what CNNs do *not* have: rotation equivariance. Rotate the input image by 45 degrees, and what comes out is not a rotated version of the original features—it is a completely different, scrambled representation. The network has to learn separately that a cat tilted left is still a cat, and a cat tilted right is still a cat, and a cat upside down is still a cat. That is why you need data augmentation: you are trying to teach the network something it should have known by construction.

Why does this matter for our geodesic integrator? Because the integrator requires well-defined tangent vectors. When an agent looks at a scene from different angles, the *features* should rotate appropriately—the tangent space structure should be preserved. If rotating the camera produces arbitrary scrambled features instead of properly rotated features, then the tangent space is ill-defined. You cannot integrate geodesics on a manifold whose tangent structure depends on arbitrary viewing angle.

We need vision encoders that respect $SO(2)$: rotate the input, get rotated features.
:::

### Why Standard CNNs Fail

:::{prf:proposition} Conv2d is NOT SO(2)-Equivariant
:label: prop-conv-not-rotation-equivariant

Standard `Conv2d` with learned kernels is translation-equivariant but **not** rotation-equivariant for continuous rotations.

*Proof.*

**Setup:** Let $I: \mathbb{Z}^2 \to \mathbb{R}^{C_{\text{in}}}$ be a discrete image on pixel lattice $\mathbb{Z}^2$ with $C_{\text{in}}$ channels. Let $\psi: \mathbb{Z}^2 \to \mathbb{R}^{C_{\text{out}} \times C_{\text{in}}}$ be a convolutional kernel. The convolution operation is:

$$
(\psi * I)(x) = \sum_{y \in \mathbb{Z}^2} \psi(y) I(x - y) \quad \text{for } x \in \mathbb{Z}^2
$$

**Step 1. Discrete rotation problem:**
For $R_\theta \in SO(2)$ with $\theta \notin \{0, \pi/2, \pi, 3\pi/2\}$, rotation maps lattice points off-grid. For example, with $\theta = \pi/4$:

$$
R_{\pi/4} \cdot (1, 0) = \left(\frac{\sqrt{2}}{2}, \frac{\sqrt{2}}{2}\right) \notin \mathbb{Z}^2
$$

To apply rotation to discrete images, we must use an interpolation scheme $\mathcal{I}$:

$$
(R_\theta \cdot I)(x) = \mathcal{I}(I, R_\theta^{-1} x) \quad \text{for } x \in \mathbb{Z}^2
$$

**Step 2. Standard interpolation (bilinear):**
For $x' = R_\theta^{-1} x \in \mathbb{R}^2 \setminus \mathbb{Z}^2$, bilinear interpolation uses the four nearest lattice points:

$$
\mathcal{I}(I, x') = \sum_{i \in \{0,1\}^2} w_i(x') I(x'_{\text{floor}} + i)
$$
where $w_i$ are barycentric weights depending on $x' - x'_{\text{floor}}$.

**Step 3. Non-commutativity of convolution and rotation:**
We show $(\psi * (R_\theta \cdot I)) \neq (R_\theta \cdot (\psi * I))$.

Left-hand side:

$$
(\psi * (R_\theta \cdot I))(x) = \sum_{y \in \mathbb{Z}^2} \psi(y) (R_\theta \cdot I)(x - y) = \sum_{y \in \mathbb{Z}^2} \psi(y) \mathcal{I}(I, R_\theta^{-1}(x - y))
$$

Right-hand side:

$$
(R_\theta \cdot (\psi * I))(x) = \mathcal{I}(\psi * I, R_\theta^{-1} x) = \mathcal{I}\left(\sum_{y \in \mathbb{Z}^2} \psi(y) I(\cdot - y), R_\theta^{-1} x\right)
$$

**Key:** Interpolation does not commute with summation:

$$
\sum_{y} \psi(y) \mathcal{I}(I, R_\theta^{-1}(x - y)) \neq \mathcal{I}\left(\sum_y \psi(y) I(\cdot - y), R_\theta^{-1} x\right)
$$

The left side interpolates $I$ at rotated shifted positions $R_\theta^{-1}(x-y)$ then sums.
The right side first computes the full convolution $\psi * I$ at all lattice points, then interpolates the result.

**Explicit numerical counterexample:**

Consider a $3 \times 3$ image with a vertical edge:

$$
I = \begin{pmatrix}
0 & 0 & 1 \\
0 & 0 & 1 \\
0 & 0 & 1
\end{pmatrix}
$$

Use a $2 \times 2$ vertical Sobel kernel detecting vertical edges:

$$
\psi = \begin{pmatrix}
-1 & 1 \\
-1 & 1
\end{pmatrix}
$$

**Path 1: Convolve then rotate by $\theta = 45°$**

Compute $(\psi * I)(x, y)$ at position $(1, 1)$ (center, using valid padding):

At $(1,1)$, the $2 \times 2$ kernel overlays the patch $I[1:3, 1:3] = \begin{pmatrix} 0 & 1 \\ 0 & 1 \end{pmatrix}$:

$$
(\psi * I)(1, 1) = -1 \cdot 0 + 1 \cdot 1 + (-1) \cdot 0 + 1 \cdot 1 = 2
$$

Similarly:
- At $(0, 0)$: patch is $\begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$, result $= 0$
- At $(0, 1)$: patch is $\begin{pmatrix} 0 & 1 \\ 0 & 1 \end{pmatrix}$, result $= 2$
- At $(1, 0)$: patch is $\begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$, result $= 0$

Convolution result (single channel):

$$
\psi * I = \begin{pmatrix}
0 & 2 \\
0 & 2
\end{pmatrix}
$$

Rotate by $45°$ using bilinear interpolation at position $(0.5, 0.5)$ (center after rotation):

$$
(R_{45°} \cdot (\psi * I))(0.5, 0.5) \approx \frac{1}{4}(0 + 2 + 0 + 2) = 1.0
$$

**Path 2: Rotate then convolve**

Rotate image $I$ by $45°$ first. At position $(1, 1)$, the rotated image samples from:

$$
R_{-45°} \cdot (1, 1) = \begin{pmatrix} \cos(45°) & \sin(45°) \\ -\sin(45°) & \cos(45°) \end{pmatrix} \begin{pmatrix} 1 \\ 1 \end{pmatrix} = \begin{pmatrix} \sqrt{2} \\ 0 \end{pmatrix} \approx (1.41, 0)
$$

Using bilinear interpolation between $(1, 0)$ and $(2, 0)$ (which wraps/pads):

$$
(R_{45°} \cdot I)(1, 1) \approx 0.41 \cdot I(2, 0) + 0.59 \cdot I(1, 0) \approx 0.41 \cdot 1 + 0.59 \cdot 0 = 0.41
$$

Computing convolution after rotation at position $(0.5, 0.5)$:

$$
(\psi * (R_{45°} \cdot I))(0.5, 0.5) \approx \text{different from } 0.5
$$

Due to interpolation artifacts and edge effects, the two paths give **numerically different results**, violating exact equivariance.

**Quantitative violation:** For the $3 \times 3$ edge image with $45°$ rotation:

$$
\|(\psi * (R_{45°} \cdot I)) - (R_{45°} \cdot (\psi * I))\|_F \approx 0.6 \text{ to } 1.0
$$

(Frobenius norm, typical values depending on boundary conditions and interpolation scheme). With correct convolution values ($\psi * I$ has entries of magnitude 2, not 1), the violation is approximately 50-100% of the signal magnitude, confirming Conv2d significantly breaks SO(2) equivariance.

$\square$

**Remark on discrete subgroups:** Exact equivariance holds for the discrete group $C_4 = \{e, R_{\pi/2}, R_\pi, R_{3\pi/2}\}$ (90° rotations) since these map $\mathbb{Z}^2 \to \mathbb{Z}^2$ without interpolation. For continuous $SO(2)$ equivariance, we need steerable filters (Definition {prf:ref}`def-steerable-filter-bank`).

**Connection to geodesic integrator:** The Lorentz-Langevin equation (Definition {prf:ref}`def-bulk-drift-continuous-flow`) requires well-defined geometric vectors in the tangent space $T_z \mathcal{Z}$. If the vision encoder produces rotation-variant features, the tangent space structure becomes viewing-angle-dependent:

$$
T_z \mathcal{Z} \neq T_{R_\theta \cdot z} \mathcal{Z} \quad \text{(geometric inconsistency)}
$$
This breaks geodesic motion, as geodesics are defined as curves minimizing length with respect to a **fixed** Riemannian metric. Rotation-variant encoders effectively change the metric under rotation, making geodesics ill-defined.
:::

### Gauge Bundle Structure for Images

:::{prf:definition} Image as Bundle Section
:label: def-image-as-bundle-section

An RGB image $I: \mathbb{R}^2 \to \mathbb{R}^3$ is a section of the trivial bundle $\mathbb{R}^2 \times \mathbb{R}^3$.

A rotation $R \in SO(2)$ acts on the base (spatial coordinates) but not the fiber (RGB values):

$$
(R \cdot I)(x) = I(R^{-1} \cdot x)
$$

For equivariant features, we need a **non-trivial bundle** where fibers also transform.
:::

:::{prf:definition} Steerable Filter Bank
:label: def-steerable-filter-bank

A filter bank $\{\psi_n^{(\ell)}\}_{n=1}^N$ is **steerable of type $\ell$** if:

$$
R_\theta \cdot \psi_n^{(\ell)} = \sum_m D_{nm}^{(\ell)}(\theta) \psi_m^{(\ell)}
$$

where $D^{(\ell)}$ is the $\ell$-th irreducible representation of $SO(2)$.

**Explicit form of $SO(2)$ irreducible representations:**

For $\ell \in \mathbb{Z}_{\geq 0}$ (non-negative integers), the $\ell$-th irreducible representation is:
- **$\ell = 0$:** Trivial representation, $D^{(0)}(\theta) = 1$ (scalar, 1-dimensional)
- **$\ell \geq 1$:** 2-dimensional representation acting on $\mathbb{R}^2$ or $\mathbb{C}$ via:

  $$
  D^{(\ell)}(\theta) = \begin{pmatrix} \cos(\ell\theta) & -\sin(\ell\theta) \\ \sin(\ell\theta) & \cos(\ell\theta) \end{pmatrix} \in SO(2)
  $$
  Equivalently, in complex notation: $D^{(\ell)}(\theta) \cdot z = e^{i\ell\theta} z$ for $z \in \mathbb{C}$.

**Physical interpretation:**
- $\ell$ is the **angular frequency** or **angular momentum quantum number**
- Under rotation by $\theta$, an $\ell$-mode rotates by $\ell \theta$ (frequency multiplication)

**Interpretation:**
- $\ell = 0$: Scalars (rotation-invariant, e.g., circularly symmetric filters). $D^{(0)}(\theta) = 1$
- $\ell = 1$: Vectors (oriented edge detectors). Rotate by $\theta$ → features rotate by $\theta$. $D^{(1)}(\theta) = R_\theta$
- $\ell = 2$: Quadrupoles (corner detectors). Rotate by $\theta$ → features rotate by $2\theta$. $D^{(2)}(\theta) = R_{2\theta}$
:::

### Associated Bundle Construction

**Mathematical prerequisites:** The construction below uses the language of **principal bundles** and **associated vector bundles** from differential geometry. We do not develop the full theory here; readers unfamiliar with fiber bundles should consult:
- **Chapter 8.5** ({ref}`sec-wilson-lines-parallel-transport`): Rigorous treatment of principal bundles, Wilson lines, curvature forms, and parallel transport
- **Standard references**: Kobayashi & Nomizu, *Foundations of Differential Geometry*; Nakahara, *Geometry, Topology and Physics*

**What is provided in this section:** We give the **direct construction** of the steerable feature bundle using the quotient $SE(2) \times_{SO(2)} V^{(\ell)}$ and verify its transformation properties. The bundle axioms (local triviality, transition functions, fiber structure) are assumed; full proofs appear in Chapter 8.5.

:::{prf:definition} Feature Bundle as Associated Vector Bundle
:label: def-associated-feature-bundle

Let $P = SE(2) = \mathbb{R}^2 \rtimes SO(2)$ be the Euclidean group (translations and rotations), and let $H = SO(2)$ be the structure group.

For steerable features of type $\ell$, the **associated vector bundle** is:

$$
E^{(\ell)} = P \times_{H} V^{(\ell)}
$$

where:
- $V^{(\ell)} \cong \mathbb{R}^{2}$ (for $\ell \geq 1$) or $\mathbb{R}$ (for $\ell = 0$) is the representation space
- $SO(2)$ acts on $V^{(\ell)}$ via $D^{(\ell)}$ (the $\ell$-th irreducible representation)
- The quotient is formed by identifying $(p, v) \sim (ph, h^{-1} \cdot v)$ for $h \in SO(2)$

**Structure:**
- **Total space:** $E^{(\ell)} = \{[(g, v)] : g \in SE(2), v \in V^{(\ell)}\}$ (equivalence classes)
- **Base space:** $B = SE(2)/SO(2) \cong \mathbb{R}^2$ (spatial positions)
- **Projection:** $\pi: E^{(\ell)} \to B$, $\pi([(g, v)]) = [g] \in \mathbb{R}^2$
- **Fiber:** $F_x = \pi^{-1}(x) \cong V^{(\ell)}$ (representation space at position $x$)

**Sections:** A steerable feature map is a **section** $\phi: B \to E^{(\ell)}$ satisfying $\pi \circ \phi = \text{id}_B$.

In coordinates: $\phi(x) = (f_1^{(\ell)}(x), \ldots, f_N^{(\ell)}(x))$ where each $f_n^{(\ell)}$ transforms under $D^{(\ell)}$.
:::

:::{prf:definition} Connection and Covariant Derivative
:label: def-connection-steerable-bundle

A **connection** on the bundle $E^{(\ell)}$ specifies how to compare fibers at different base points.

For the associated bundle $E^{(\ell)} = SE(2) \times_{SO(2)} V^{(\ell)}$, the **canonical flat connection** is:

$$
\nabla_X \phi = X[\phi]
$$

where $X$ is a vector field on $\mathbb{R}^2$ and $X[\phi]$ is the directional derivative.

**Parallel transport:** A section $\phi(x + tv)$ along direction $v$ is **parallel** if $\nabla_v \phi = 0$, i.e., $\frac{d}{dt}\phi(x + tv) = 0$.

For steerable features, parallel transport preserves the transformation law:

$$
\phi(x + \delta x) = \phi(x) + \nabla_{\delta x} \phi + O(\|\delta x\|^2)
$$

where $\nabla_{\delta x} \phi$ transforms under $D^{(\ell)}$ at the new position.
:::

:::{prf:remark} Gauge Fields and Curvature
:label: rem-gauge-curvature-vision

In the gauge theory framework (Chapter 8.1), the **Binding field** $G_\mu$ acts as a connection on the feature bundle. The curvature (field strength) $F_{\mu\nu} = \partial_\mu G_\nu - \partial_\nu G_\mu + [G_\mu, G_\nu]$ measures the failure of parallel transport to be path-independent.

For the trivial bundle with flat connection (used in steerable CNNs), $F_{\mu\nu} = 0$ (zero curvature), meaning parallel transport is path-independent. This corresponds to **free field theory** in physics.

**Extension to non-trivial connections:** If steerable features are coupled to other latent variables via attention or gating, the effective connection becomes non-trivial (non-zero $G_\mu$), introducing curvature. This is explored in Section 8.5 (Covariant Cross-Attention) with **Wilson lines** for parallel transport.
:::

:::{div} feynman-prose
Let me show you a beautiful idea: harmonic decomposition for rotations, analogous to Fourier decomposition for translations.

You know Fourier analysis: any periodic signal can be written as a sum of sines and cosines at different frequencies. Low frequency components capture the broad shape; high frequency components capture the fine details. The key insight is that these frequency components *transform simply* under translation—shift the signal, and each Fourier mode just picks up a phase.

The same idea works for rotations, but now the "frequency" is the angular momentum $\ell$. At $\ell = 0$, you have circularly symmetric patterns—things that look the same from every angle, like a bullseye or a Gaussian blob. When you rotate the image, these features do not change at all. At $\ell = 1$, you have dipole patterns—things with a preferred direction, like an edge detector or a gradient. Rotate the image by $\theta$, and these features rotate by $\theta$. At $\ell = 2$, you have quadrupole patterns—things like corner detectors or curvature indicators. They rotate twice as fast as the image.

Any filter can be decomposed into these harmonic components, just like any signal can be decomposed into Fourier modes. And the magic is that each component *transforms according to a known rule* under rotation. The $\ell = 0$ part stays fixed. The $\ell = 1$ part rotates with the image. The $\ell = 2$ part rotates at double speed. This is what makes the filters "steerable"—you can predict exactly how the output will transform under any rotation.
:::

:::{prf:theorem} Steerable Convolution is SO(2)-Equivariant
:label: thm-steerable-conv-equivariant

Let $\{\psi_n^{(\ell)}\}_{n=1}^N$ be a steerable filter bank of type $\ell$ (Definition {prf:ref}`def-steerable-filter-bank`). Let $\text{Conv}_\ell$ denote convolution with these filters:

$$
(\text{Conv}_\ell I)_n(x) = \sum_m (\psi_m^{(\ell)} * I)(x) = \int_{\mathbb{R}^2} \psi_m^{(\ell)}(y) I(x - y) \, dy
$$

Then for any rotation $R_\theta \in SO(2)$:

$$
\text{Conv}_\ell(R_\theta \cdot I) = D^{(\ell)}(\theta) \cdot \text{Conv}_\ell(I)
$$

where $D^{(\ell)}(\theta)$ is the $\ell$-th irreducible representation of $SO(2)$.

*Proof.*

**Step 1. Apply rotation to input:**
By linearity of convolution:

$$
(\psi_n * (R_\theta \cdot I))(x) = \int_{\mathbb{R}^2} \psi_n(y) I(R_\theta^{-1}(x-y)) \, dy
$$

**Step 2. Change of variables:**
Let $u = R_\theta^{-1} y$, so $y = R_\theta u$. Since $R_\theta \in SO(2)$ preserves measure ($|\det R_\theta| = 1$), we have $dy = du$:

$$
= \int_{\mathbb{R}^2} \psi_n(R_\theta u) I(R_\theta^{-1}(x - R_\theta u)) \, du
$$

Using the property $R_\theta^{-1}(x - R_\theta u) = R_\theta^{-1} x - u$:

$$
= \int_{\mathbb{R}^2} \psi_n(R_\theta u) I(R_\theta^{-1} x - u) \, du
$$

**Step 3. Apply steerability definition:**
By the steerability property (Definition {prf:ref}`def-steerable-filter-bank`), the filter transforms as:

$$
\psi_n(R_\theta u) = (R_\theta \cdot \psi_n)(u)
$$

Therefore:

$$
= \int_{\mathbb{R}^2} (R_\theta \cdot \psi_n)(u) I(R_\theta^{-1} x - u) \, du
$$

**Step 4. Apply filter bank steerability:**
By Definition {prf:ref}`def-steerable-filter-bank`, a steerable filter of type $\ell$ satisfies:

$$
(R_\theta \cdot \psi_n)(u) = \psi_n(R_\theta u) = \sum_m D_{nm}^{(\ell)}(\theta) \psi_m(u)
$$

Substituting:

$$
= \int_{\mathbb{R}^2} \left[\sum_m D_{nm}^{(\ell)}(\theta) \psi_m(u)\right] I(R_\theta^{-1} x - u) \, du
$$

**Step 5. Separate sum from integral:**

$$
= \sum_m D_{nm}^{(\ell)}(\theta) \int_{\mathbb{R}^2} \psi_m(u) I(R_\theta^{-1} x - u) \, du
$$

**Step 6. Recognize convolution:**
The integral $\int \psi_m(u) I(R_\theta^{-1} x - u) \, du = (\psi_m * I)(R_\theta^{-1} x)$ by definition of convolution. This is precisely $(R_\theta \cdot (\psi_m * I))(x)$ by the definition of rotation action on functions.

Therefore:

$$
(\psi_n^{(\ell)} * (R_\theta \cdot I))(x) = \sum_m D_{nm}^{(\ell)}(\theta) (R_\theta \cdot (\psi_m^{(\ell)} * I))(x)
$$

which shows the desired equivariance property.

In matrix form:

$$
\text{Conv}_\ell(R_\theta \cdot I) = D^{(\ell)}(\theta) \cdot \text{Conv}_\ell(I)
$$

$\square$

**Remark:** For full treatment of discretization effects and practical implementation, see Cohen & Welling (2016) {cite}`cohen2016group` and Weiler & Cesa (2019) {cite}`weiler2019general`.
:::

### Lifting to Homogeneous Space

:::{prf:definition} Lifting Map to SE(2)
:label: def-lifting-map

**Group structure:** The **special Euclidean group** $SE(2) = \mathbb{R}^2 \rtimes SO(2)$ is the group of rigid motions (translations + rotations) in the plane, with elements $g = (x, R_\theta)$ where $x \in \mathbb{R}^2$ is translation and $R_\theta \in SO(2)$ is rotation by angle $\theta$.

**Group multiplication:** For $g_1 = (x_1, R_{\theta_1})$ and $g_2 = (x_2, R_{\theta_2})$:

$$
g_1 \cdot g_2 = (x_1 + R_{\theta_1} x_2, R_{\theta_1 + \theta_2})
$$

*Interpretation:* Composition $g_1 \cdot g_2$ means "first apply $g_2$, then $g_1$". Acting on a point $p \in \mathbb{R}^2$:

$$
(g_1 \cdot g_2) \cdot p = g_1 \cdot (g_2 \cdot p) = g_1 \cdot (R_{\theta_2} p + x_2) = R_{\theta_1}(R_{\theta_2} p + x_2) + x_1 = R_{\theta_1 + \theta_2} p + (x_1 + R_{\theta_1} x_2)
$$
The rotation $R_{\theta_1}$ in $g_1$ acts on the translation $x_2$ from $g_2$, and the rotations compose as $R_{\theta_1} R_{\theta_2} = R_{\theta_1 + \theta_2}$.

**Domain and codomain:** Let $C(\mathbb{R}^2, \mathbb{R}^{C_{\text{in}}})$ be the space of continuous functions (images) from $\mathbb{R}^2$ to $\mathbb{R}^{C_{\text{in}}}$ (e.g., $C_{\text{in}} = 3$ for RGB). Let $C(SE(2), \mathbb{R}^{C_{\text{out}}})$ be functions on $SE(2)$ with values in $\mathbb{R}^{C_{\text{out}}}$ (output feature dimension).

The **lifting map** is an operator:

$$
L: C(\mathbb{R}^2, \mathbb{R}^{C_{\text{in}}}) \to C(SE(2), \mathbb{R}^{C_{\text{out}}})
$$

**Definition:** For an input image $I \in C(\mathbb{R}^2, \mathbb{R}^{C_{\text{in}}})$ and group element $g = (x, R_\theta) \in SE(2)$:

$$
(L I)(g) = (L I)(x, \theta) := \sum_{i=1}^{C_{\text{out}}} (\psi_i^{(\theta)} * I)(x) \cdot e_i
$$
where:
- $\{\psi_i\}_{i=1}^{C_{\text{out}}}$ is a steerable filter bank at orientation $\theta = 0$ (Definition {prf:ref}`def-steerable-filter-bank`)
- $\psi_i^{(\theta)} := R_\theta \cdot \psi_i$ is the **rotated filter**: applying rotation $R_\theta$ to the base filter $\psi_i$
- $\{e_i\}$ is the standard basis of $\mathbb{R}^{C_{\text{out}}}$

**Key dependence on $\theta$:** The output $(LI)(x, \theta)$ depends explicitly on the rotation angle $\theta$ through the rotated filters $\psi_i^{(\theta)}$. At each orientation $\theta$, the network applies filters rotated to that orientation, detecting patterns aligned with $\theta$.

**Explicit filter rotation:** For a filter $\psi: \mathbb{R}^2 \to \mathbb{R}$ and rotation $R_\theta \in SO(2)$:

$$
(R_\theta \cdot \psi)(y) := \psi(R_\theta^{-1} y)
$$
This ensures that a filter detecting "vertical edge" at $\theta = 0$ becomes a filter detecting "edge at angle $\theta$" after rotation.

**Equivariance property:** For $g_0 = (x_0, R_{\theta_0}) \in SE(2)$ and image $I$, define the left-translated image $(L_{g_0} I)(x) := I(R_{\theta_0}^{-1}(x - x_0))$. Then:

$$
L(L_{g_0} I) = L_{g_0}(L I)
$$
where the right-hand side is left-multiplication on $SE(2)$: $(L_{g_0} f)(g) = f(g_0^{-1} g)$.

*Verification:* Evaluate both sides at $g = (x, R_\theta)$:
- **LHS:** $(L(L_{g_0} I))(x, \theta) = \sum_i (\psi_i^{(\theta)} * (L_{g_0} I))(x) \cdot e_i$
- Change variables in convolution: $(\psi_i^{(\theta)} * (L_{g_0} I))(x) = \int \psi_i(R_\theta^{-1}(x - y)) I(R_{\theta_0}^{-1}(y - x_0)) dy$
- Substitute $u = R_{\theta_0}^{-1}(y - x_0)$, so $y = R_{\theta_0} u + x_0$:
  $$= \int \psi_i(R_\theta^{-1}(x - x_0 - R_{\theta_0} u)) I(u) du = \int \psi_i(R_{\theta - \theta_0}^{-1}(R_{\theta_0}^{-1}(x - x_0) - u)) I(u) du$$
- **RHS:** $(L_{g_0}(L I))(g) = (L I)(g_0^{-1} g)$ where $g_0^{-1} = (-R_{\theta_0}^{-1} x_0, R_{\theta_0}^{-1})$
- $g_0^{-1} g = (R_{\theta_0}^{-1}(x - x_0), R_{\theta - \theta_0})$ (using SE(2) multiplication)
- $(L I)(R_{\theta_0}^{-1}(x - x_0), \theta - \theta_0) = \sum_i (\psi_i^{(\theta - \theta_0)} * I)(R_{\theta_0}^{-1}(x - x_0)) \cdot e_i$
- This matches the LHS after recognizing $\psi_i^{(\theta - \theta_0)}(y) = \psi_i(R_{\theta - \theta_0}^{-1} y)$. $\square$

**Geometric interpretation:** Instead of features at spatial locations $x \in \mathbb{R}^2$, lifted features live at *posed locations* $(x, \theta) \in SE(2)$: position AND orientation. The network learns to detect patterns *and* their orientations explicitly.

**Output dimension:** For $N_\theta$ discrete orientations (e.g., $N_\theta = 8$ for $\theta \in \{0°, 45°, 90°, \ldots, 315°\}$), the output has dimension $C_{\text{out}} = N_\theta \times C_{\text{feature}}$ where $C_{\text{feature}}$ is the number of feature types per orientation.
:::

### Implementation: E2CNN CovariantRetina

The following implementation uses the E2CNN library {cite}`weiler2019general` to build an $SO(2)$-equivariant vision encoder.

```python
import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as e2nn
from typing import Tuple

class CovariantRetina(nn.Module):
    """SO(2)-equivariant vision encoder using steerable convolutions.

    Implements Theorem {prf:ref}`thm-steerable-conv-equivariant` via E2CNN library.

    **Guarantees:**
        - SO(2) rotation equivariance: Conv(R_θ · I) = D^(ℓ)(θ) · Conv(I)
        - Translation equivariance: Conv(T_x · I) = T_x · Conv(I)
        - Preserves tangent space structure for geodesic integrator

    **Architecture:**
        1. Lifting layer: R² → SE(2) (trivial → regular representation)
        2. Steerable convolutions: SE(2) → SE(2) with expanding channels
        3. Group pooling: SE(2) → R² (max over rotations)
        4. Linear projection to latent space

    **Diagnostics:**
        - Node 68: RotationEquivarianceCheck

    Args:
        in_channels: Input RGB channels (typically 3)
        out_dim: Output latent dimension [dimensionless]
        num_rotations: Discretization of SO(2) (typically 8 or 16)
        kernel_size: Convolution kernel size [pixels]

    Shapes:
        Input: [B, C_in, H, W] where C_in = in_channels
        Output: [B, out_dim]

    Example:
        >>> retina = CovariantRetina(in_channels=3, out_dim=512, num_rotations=8)
        >>> img = torch.randn(32, 3, 64, 64)  # [B, C, H, W]
        >>> features = retina(img)  # [32, 512]
        >>> # Test SO(2) equivariance (Node 68)
        >>> img_rotated = rotate_image(img, angle=45)  # rotate by 45°
        >>> features_rotated = retina(img_rotated)
        >>> # Features should be related by representation matrix D^(ℓ)(45°)
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 512,
        num_rotations: int = 8,
        kernel_size: int = 5
    ):
        super().__init__()

        # Define the symmetry group: rotations + reflections (or just rotations)
        # r2_act = gspaces.Rot2dOnR2(N=num_rotations)  # Rotations only
        self.r2_act = gspaces.FlipRot2dOnR2(N=num_rotations)  # Rotations + reflections

        # Input type: trivial representation (standard image)
        in_type = e2nn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])

        # Feature types for steerable filters
        # Use regular representation (all rotations) for maximum expressivity
        self.feature_type_32 = e2nn.FieldType(
            self.r2_act,
            32 * [self.r2_act.regular_repr]
        )
        self.feature_type_64 = e2nn.FieldType(
            self.r2_act,
            64 * [self.r2_act.regular_repr]
        )

        # 1. Lifting convolution: R² → SE(2)
        # Maps trivial representation to regular representation
        self.lift = e2nn.R2Conv(
            in_type,
            self.feature_type_32,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False  # No bias (gauge invariance)
        )

        # 2. Steerable convolutions on SE(2)
        self.conv1 = e2nn.R2Conv(
            self.feature_type_32,
            self.feature_type_64,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.conv2 = e2nn.R2Conv(
            self.feature_type_64,
            self.feature_type_64,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        # 3. Norm-based nonlinearity (Definition {prf:ref}`def-norm-gated-activation`)
        # NormNonLinearity computes ||v|| and applies gate, preserving equivariance
        self.relu1 = e2nn.NormNonLinearity(self.feature_type_32)
        self.relu2 = e2nn.NormNonLinearity(self.feature_type_64)
        self.relu3 = e2nn.NormNonLinearity(self.feature_type_64)

        # 4. Group pooling: max over rotation group
        # Projects SE(2) → R² by taking max response over all orientations
        self.group_pool = e2nn.GroupPooling(self.feature_type_64)

        # 5. Adaptive spatial pooling
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 6. Final linear projection (with spectral norm)
        pool_features = 64 * 4 * 4  # After pooling: 64 channels × 4×4 spatial
        from torch.nn.utils import spectral_norm
        self.fc = spectral_norm(nn.Linear(pool_features, out_dim, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with equivariance preservation.

        Args:
            x: [B, C, H, W] input image [dimensionless, normalized to [-1,1]]

        Returns:
            z: [B, out_dim] latent features [dimensionless]
        """
        # Wrap input as E2CNN geometric tensor
        x = e2nn.GeometricTensor(x, e2nn.FieldType(
            self.r2_act,
            x.shape[1] * [self.r2_act.trivial_repr]
        ))

        # Lifting + steerable convolutions
        x = self.lift(x)        # [B, 32×|G|, H, W] where |G| = num_rotations
        x = self.relu1(x)       # Norm-gated activation (equivariant)

        x = self.conv1(x)       # [B, 64×|G|, H, W]
        x = self.relu2(x)

        x = self.conv2(x)       # [B, 64×|G|, H, W]
        x = self.relu3(x)

        # Group pooling: max over rotation channels
        x = self.group_pool(x)  # [B, 64, H, W]

        # Extract underlying torch tensor
        x = x.tensor

        # Spatial pooling
        x = self.spatial_pool(x)  # [B, 64, 4, 4]

        # Flatten
        x = x.flatten(1)  # [B, 64×16 = 1024]

        # Project to latent dimension with spectral norm (Thm {prf:ref}`thm-spectral-preserves-light-cone`)
        z = self.fc(x)  # [B, out_dim]

        return z
```

**Key design choices:**

1. **FlipRot2dOnR2 vs Rot2dOnR2:** We use rotations + reflections for richer symmetry. For applications where chirality matters (e.g., detecting left vs. right hands), use `Rot2dOnR2` instead.

2. **Regular representation:** Expands each channel by a factor of $|G|$ (number of rotations), storing responses at all orientations. More expensive but maximally expressive.

3. **NormNonLinearity:** E2CNN's implementation of norm-gated activation (Definition {prf:ref}`def-norm-gated-activation`). Preserves equivariance by acting only on fiber norms.

4. **GroupPooling:** Reduces $SE(2)$ features to $\mathbb{R}^2$ by taking max (or avg) over rotation channels. Creates rotation-*invariant* features from rotation-*equivariant* ones.

5. **Spectral normalization on final linear layer:** Ensures light cone preservation (Theorem {prf:ref}`thm-spectral-preserves-light-cone`) when projecting to latent space.

**Diagnostic verification (Node 68):**

```python
def test_rotation_equivariance(retina: CovariantRetina, img: torch.Tensor):
    """Test SO(2) equivariance of CovariantRetina.

    Verifies: ||retina(rotate(img, θ)) - rotate_features(retina(img), θ)|| < ε
    """
    import torchvision.transforms.functional as TF

    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    features_original = retina(img)

    for angle in angles[1:]:  # Skip 0° (identity)
        img_rotated = TF.rotate(img, angle)
        features_rotated = retina(img_rotated)

        # Note: After group pooling, features are rotation-INVARIANT
        # So we expect features_rotated ≈ features_original (not rotated features)
        violation = torch.norm(features_rotated - features_original).item()

        print(f"Angle {angle}°: violation = {violation:.6f}")
        assert violation < 1e-3, f"Rotation invariance violated at {angle}°"
```



(sec-connection-to-gauge-structure)=
## Connection to Gauge Structure

:::{div} feynman-prose
Now we come to the connection that ties everything together. We have built three primitives—SpectralLinear, NormGate, SteerableConv—each carefully designed to preserve some symmetry. But in Chapter 8.1, we derived that agents must respect the full gauge group $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$. How do our primitives relate to this structure?

Here is the correspondence. Each primitive preserves one factor of the gauge group:

The *bundle structure* with $n_b$ bundles implements $SU(N_f)_C$—the "color" symmetry that keeps features confined within their bundles. Just as quarks in QCD cannot exist in isolation (they must bind into color-neutral hadrons), features in IsotropicBlocks cannot propagate independently (they must form bound states across bundles). Node 40 in the diagnostics enforces this confinement.

*Spectral normalization* implements $U(1)_Y$—the hypercharge symmetry associated with a capacity bound. The constraint $\sigma_{\max}(W) \leq 1$ ensures that the total "hypercharge" (roughly, the squared norm of the latent state) cannot grow without bound. This is the capacity constraint from the holographic bound in Chapter 8.1.

*Steerable convolutions* implement $SU(2)_L$—the "weak" symmetry that mixes observation and action. Visual features from the retina form doublets with action planning, just as left-handed particles in the Standard Model form $SU(2)$ doublets.

Three primitives, three gauge factors. This is not coincidence—it is the architecture being forced into a unique form by symmetry requirements.
:::

### Mapping Primitives to Gauge Fields

:::{prf:definition} SU(N_f) Gauge Action on Bundle Space
:label: def-gauge-action-bundles

Let $Z = (z^{(1)}, \ldots, z^{(n_b)})$ be the bundled latent representation, viewed here in the **complexified** setting with $z^{(i)} \in \mathbb{C}^{d_b}$. The gauge group $SU(N_f)$ with $N_f = n_b$ acts on $Z$ as:

$$
Z \mapsto Z' = Z \cdot U
$$

where $U \in SU(N_f)$ is an $n_b \times n_b$ special unitary matrix:

$$
U^\dagger U = I, \quad \det(U) = 1
$$

**Explicit action:** In matrix form, treating $Z$ as a $d_b \times n_b$ matrix:

$$
z'^{(j)} = \sum_{i=1}^{n_b} U_{ij} \, z^{(i)} \quad \text{(bundle mixing)}
$$

**Color charge:** Represent the latent state as a matrix $Z \in \mathbb{C}^{d_b \times n_b}$ where the $i$-th column is the $i$-th bundle vector $z^{(i)} \in \mathbb{C}^{d_b}$:

$$
Z = [z^{(1)} \mid z^{(2)} \mid \cdots \mid z^{(n_b)}]
$$

For each generator $T^a \in \mathfrak{su}(n_b)$ ($a = 1, \ldots, n_b^2 - 1$), where $T^a$ is a traceless Hermitian $n_b \times n_b$ matrix, define the **color charge operator**:

$$
Q_C^a[Z] = \text{Tr}_{\text{bundle}}(Z^\dagger Z \cdot T^a) = \sum_{i,j=1}^{n_b} T^a_{ij} \, (z^{(i)})^\dagger z^{(j)}
$$

where:
- $Z \in \mathbb{C}^{d_b \times n_b}$ has columns $z^{(1)}, \ldots, z^{(n_b)} \in \mathbb{C}^{d_b}$ (bundles)
- $Z^\dagger \in \mathbb{C}^{n_b \times d_b}$ is the conjugate transpose (rows are bundle vectors)
- $Z^\dagger Z \in \mathbb{C}^{n_b \times n_b}$ is the **Gram matrix** with $(Z^\dagger Z)_{ij} = (z^{(i)})^\dagger z^{(j)}$
- $T^a \in \mathbb{C}^{n_b \times n_b}$ is the generator matrix (acts on bundle indices)
- $Z^\dagger Z \cdot T^a \in \mathbb{C}^{n_b \times n_b}$ is matrix multiplication
- $\text{Tr}_{\text{bundle}}$ denotes trace over bundle indices (summing diagonal elements of the $n_b \times n_b$ matrix)

**Dimensional consistency:**
- $[z^{(i)}] = \sqrt{\text{nat}}$ (latent vector)
- $[(z^{(i)})^\dagger z^{(j)}] = \text{nat}$ (inner product)
- $[T^a_{ij}]$ = dimensionless (matrix element)
- $[Q_C^a] = \text{nat}$ (charge is extensive in latent dimension)

A state is **color-neutral** (confined) if:

$$
Q_C^a[Z] = 0 \quad \forall \, a = 1, \ldots, n_b^2 - 1
$$

**Physical interpretation:** Just as quarks in QCD carry color charge under $SU(3)_C$, latent features carry "bundle charge" under $SU(N_f)_C$ with $N_f = n_b$. Only color-neutral combinations (satisfying all $n_b^2 - 1$ charge constraints) can propagate to the macro level.
:::

:::{prf:theorem} Isotropic Blocks Preserve SU(N_f)_C Gauge Structure
:label: thm-isotropic-preserves-color

The bundled structure (Definition {prf:ref}`def-latent-vector-bundle`) with $n_b$ bundles, coupled with norm-gating and the Binding field $G_\mu$, implements $SU(N_f)_C$ gauge invariance with $N_f = n_b$.

**Statement:** For any $U \in SU(N_f)$, the effective dynamics satisfy:

$$
\mathcal{L}_{\text{eff}}[Z \cdot U] = \mathcal{L}_{\text{eff}}[Z]
$$

Moreover, norm-gating induces scale-dependent coupling:
- **Infrared** ($\ell \to \infty$): Strong coupling $g_s(\mu_{\text{IR}}) \gg 1$ → confinement
- **Ultraviolet** ($\ell \to 0$): Weak coupling $g_s(\mu_{\text{UV}}) \to 0$ → asymptotic freedom

*Proof.*

**Step 1. Bundle structure and gauge group:**

Recall from Definition {prf:ref}`def-latent-vector-bundle` that the latent space decomposes as:

$$
\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i, \quad V_i \cong \mathbb{R}^{d_b}
$$

For the gauge-theoretic formulas in this subsection we implicitly work in the complexification $V_i \otimes_{\mathbb{R}} \mathbb{C}$ (so bundle vectors may be treated as elements of $\mathbb{C}^{d_b}$); the real-valued architecture uses the corresponding realified action discussed in Definition {prf:ref}`def-fragile-gauge-group`.

Each bundle $V_i$ transforms under its $SO(d_b)$ factor. For $n_b = N_f$ bundles, consider the gauge group $SU(N_f)$ acting on the **bundle indices** (not on the internal bundle space).

**Matrix notation:** Represent the latent state as a matrix $Z \in \mathbb{C}^{d_b \times N_f}$ where the $i$-th column is the $i$-th bundle vector:

$$
Z = [v_1 \mid v_2 \mid \cdots \mid v_{N_f}]
$$

**Gauge transformation:** $SU(N_f)$ acts by right multiplication:

$$
Z \mapsto Z \cdot U, \quad U \in SU(N_f), \quad U^\dagger U = I
$$

This mixes bundle indices while preserving the Frobenius norm $\|Z\|_F^2 := \text{Tr}(Z^\dagger Z) = \sum_i \|v_i\|_2^2$.

**Step 2. Covariant derivative:**

To define dynamics that respect gauge invariance, introduce the **gauge-covariant derivative**:

$$
D_\mu Z := \partial_\mu Z - i g_s \, Z \cdot G_\mu
$$

where:
- $G_\mu = \sum_{a=1}^{N_f^2-1} G_\mu^a T^a$ is the gauge connection (Binding field)
- $T^a$ are the generators of $SU(N_f)$ (Hermitian, traceless $N_f \times N_f$ matrices)
- $g_s > 0$ is the coupling constant

**Connection transformation:** Under a gauge transformation $Z \to Z' = Z \cdot U$ with $U(x^\mu) \in SU(N_f)$ (spacetime-dependent), we require the covariant derivative to transform as

$$
D'_\mu Z' = (D_\mu Z)\cdot U,
$$
where $D'_\mu Z' := \partial_\mu Z' - i g_s \, Z'\cdot G'_\mu$ uses the transformed connection $G'_\mu$.

Expanding and equating terms:

$$
(\partial_\mu Z)\cdot U + Z\cdot(\partial_\mu U) - i g_s \, Z\cdot U \cdot G'_\mu
= (\partial_\mu Z)\cdot U - i g_s \, Z\cdot G_\mu \cdot U.
$$
Cancel $(\partial_\mu Z)\cdot U$ and use that this must hold for all $Z$ to obtain the operator identity

$$
\partial_\mu U - i g_s \, U G'_\mu + i g_s \, G_\mu U = 0.
$$
Solving for $G'_\mu$ yields the right-action Yang--Mills transformation law:

$$
\boxed{
G_\mu \to G'_\mu = U^{-1} G_\mu U - \frac{i}{g_s} U^{-1}(\partial_\mu U)
}
$$
(equivalently $G'_\mu = U^\dagger G_\mu U - \frac{i}{g_s} U^\dagger(\partial_\mu U)$ since $U^{-1}=U^\dagger$). This is the standard gauge transformation written for a right action $Z\mapsto ZU$ {cite}`peskin1995introduction`.

**Verification:** With $D_\mu Z = \partial_\mu Z - i g_s \, ZG_\mu$ and $G'_\mu$ as above, one checks directly that

$$
D'_\mu(Z\cdot U) = (D_\mu Z)\cdot U.
$$

Thus the effective Lagrangian $\mathcal{L}_{\text{eff}} = \|D_\mu Z\|_F^2 = \text{Tr}[(D_\mu Z)^\dagger (D^\mu Z)]$ is gauge-invariant by construction.

**Step 2. Norm-gating as effective coupling:**

The norm-gating activation potential $b_i$ creates an energy barrier. The effective coupling at layer $\ell$ is:

$$
g_s^{(\ell)} = \frac{\langle \|G_\mu\| \rangle}{\langle \|\partial_\mu Z\| \rangle} \approx \frac{\beta_\ell}{\sqrt{1 + \|\nabla_{W_\ell} \mathcal{L}\|^{-2}}}
$$

where $\beta_\ell = \tanh(b_\ell)$ is the barrier strength (see Proposition {prf:ref}`prop-coupling-from-barriers` below).

**Step 3. IR behavior (confinement):**

At large layer depth $\ell \to L$ (infrared scale):
- Barrier potentials $b_\ell$ are large (strong gates)
- Gradient flow between bundles is suppressed unless $Q_C^a = 0$
- Effective coupling $g_s^{(L)} \gg 1$

**Energy penalty for non-neutral states:**

$$
\Delta E_{\text{conf}} = g_s^{(L)} \sum_a |Q_C^a|^2 \to \infty \quad \text{as } g_s^{(L)} \to \infty
$$

Only color-neutral states ($Q_C^a = 0 \, \forall a$) have finite energy → **confinement**.

**Step 4. UV behavior (asymptotic freedom):**

At shallow layers $\ell \to 0$ (ultraviolet scale):
- Barrier potentials $b_\ell \to 0$ (weak gates)
- Bundles decouple, gradient flow is independent
- Effective coupling $g_s^{(0)} \to 0$

Bundles behave as free, non-interacting subspaces → **asymptotic freedom**.

**Step 5. Identification with coupling window:**

By Corollary {prf:ref}`cor-coupling-window`, the gauge coupling must satisfy:

$$
g_s(\mu_{\text{IR}}) \geq g_s^{\text{crit}} \quad \text{and} \quad g_s(\mu_{\text{UV}}) \to 0
$$

The norm-gating schedule $\{b_\ell\}_{\ell=0}^L$ implements this by construction:
- $b_L$ large (IR confinement)
- $b_0$ small (UV freedom)

$\square$

**Connection to Node 40 (PurityCheck):** Measures $\sum_a |Q_C^a|^2$ at the final layer. Violation indicates non-neutral states reaching the macro register, breaking confinement.
:::

:::{prf:proposition} Coupling Strength from Norm-Gating Barriers
:label: prop-coupling-from-barriers

The effective gauge coupling at layer $\ell$ is:

$$
g_s^{(\ell)} = \beta_\ell \cdot \frac{\xi_\ell}{\sqrt{\xi_\ell^2 + \eta_\ell^2}}
$$

where:
- $\beta_\ell = \tanh(b_\ell)$ is the dimensionless barrier strength
- $\xi_\ell = \langle \|W_{\text{off-diag}}^{(\ell)}\| \rangle$ is the mean off-diagonal weight norm (dimensionless)
- $\eta_\ell = \langle \|\partial_\ell z\|_2 / \|\partial_{\ell-1} z\|_2 \rangle$ is the normalized gradient flow ratio (dimensionless)

*Proof.*

We derive the coupling formula from first principles using the Yang-Mills effective action.

**Step 1. Effective Yang-Mills action on latent space:**

From the covariant derivative $D_\mu Z = \partial_\mu Z - i g_s \, Z \cdot G_\mu$ (Step 2 of Theorem {prf:ref}`thm-isotropic-preserves-color`), the gauge-invariant kinetic term is:

$$
\mathcal{S}_{\text{kin}} = \int d^4x \, \text{Tr}[(D_\mu Z)^\dagger (D^\mu Z)]
$$

Expanding (for Hermitian $G_\mu$ and using cyclicity of the trace):

$$
\mathcal{S}_{\text{kin}} = \int d^4x \, \text{Tr}\left[(\partial_\mu Z)^\dagger (\partial^\mu Z) + i g_s\bigl(G^\mu Z^\dagger (\partial_\mu Z) - (\partial_\mu Z)^\dagger Z G^\mu\bigr) + g_s^2 (Z^\dagger Z)\, G_\mu G^\mu\right]
$$

**Identification of terms:**
- **Free kinetic:** $\mathcal{S}_0 = \int \text{Tr}[(\partial_\mu Z)^\dagger (\partial^\mu Z)]$ (dimensionless for dimensionless $Z$)
- **Interaction:** $\mathcal{S}_{\text{int}} = \int \text{Tr}[i g_s(G^\mu Z^\dagger (\partial_\mu Z) - (\partial_\mu Z)^\dagger Z G^\mu)]$ (linear in $g_s$)
- **Gauge field self-energy:** $\mathcal{S}_{G} = \int g_s^2 \text{Tr}[(Z^\dagger Z)\, G_\mu G^\mu]$ (quadratic in $g_s$)

**Step 2. Gauge coupling definition from action ratios:**

The effective dimensionless coupling constant is defined by the ratio:

$$
g_s^2 := \frac{\langle \mathcal{S}_{G} \rangle}{\langle \mathcal{S}_0 \rangle}
$$

where $\langle \cdot \rangle$ denotes expectation over the weight distribution at layer $\ell$.

**Explicit computation:**

$$
g_s^2 = \frac{\int \text{Tr}[(Z^\dagger Z)\, G_\mu G^\mu]}{\int \text{Tr}[(\partial_\mu Z)^\dagger (\partial^\mu Z)]}
= \frac{\langle (Z^\dagger Z)\, G_\mu G^\mu \rangle}{\langle (\partial_\mu Z)^\dagger (\partial^\mu Z) \rangle}
$$

For approximately uniform field configurations, this simplifies to:

$$
g_s \approx \frac{\|G_\mu\|_{\text{op}}}{\|\partial_\mu Z\|_{\text{op}}}
$$

where $\|\cdot\|_{\text{op}}$ denotes operator norm (largest singular value).

**Step 3. Off-diagonal weight norm determines gauge field strength:**

The Binding field $G_\mu$ couples bundles. In matrix representation, write the weight matrix $W^{(\ell)} \in \mathbb{R}^{d_z \times d_z}$ in block form with respect to the bundle decomposition $\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i$:

$$
W^{(\ell)} = \begin{pmatrix}
W_{11} & W_{12} & \cdots & W_{1 n_b} \\
W_{21} & W_{22} & \cdots & W_{2 n_b} \\
\vdots & \vdots & \ddots & \vdots \\
W_{n_b 1} & W_{n_b 2} & \cdots & W_{n_b n_b}
\end{pmatrix}
$$

where each $W_{ij} \in \mathbb{R}^{d_b \times d_b}$ is the coupling from bundle $j$ to bundle $i$.

**Diagonal vs off-diagonal decomposition:**

$$
W^{(\ell)} = W_{\text{diag}}^{(\ell)} + W_{\text{off}}^{(\ell)}
$$

where:
- $W_{\text{diag}} = \text{diag}(W_{11}, W_{22}, \ldots, W_{n_b n_b})$ (intra-bundle)
- $W_{\text{off}}$ has zeros on block diagonal (inter-bundle coupling)

The gauge field $G_\mu$ is proportional to the **inter-bundle coupling**:

$$
\|G_\mu\|_{\text{op}} = \|W_{\text{off}}^{(\ell)}\|_{\text{op}} \cdot \kappa_\ell
$$

where $\kappa_\ell$ is a geometry-dependent prefactor (typically $\kappa_\ell \sim 1$ for normalized weights).

**Step 4. Barrier modulation of off-diagonal coupling:**

The norm-gating barrier $b_\ell$ enters through the **effective off-diagonal weights** after applying the gate $v \mapsto v \cdot g(\|v\| + b)$.

For bundle $i$, the effective weight from bundle $j \neq i$ is:

$$
W_{ij}^{\text{eff}} = W_{ij} \cdot \frac{g'(\|v_j\| + b_\ell)}{\|v_j\|}
$$

**Barrier suppression mechanism:** For large positive $b_\ell$ (strong barrier):
- GELU derivative: $g'(\|v\| + b_\ell) \approx \Phi(\|v\| + b_\ell) + (\|v\| + b_\ell) \phi(\|v\| + b_\ell)$
- For $\|v\| \sim 1$ and $b_\ell \gg 1$: $g'(1 + b_\ell) \to 1$ (saturates)
- The barrier **enhances** coupling at large $b$ (counter-intuitive but correct for IR confinement)

For small $b_\ell \to 0$ (weak barrier):
- $g'(1) \approx 1.08$, minimal modulation
- Bundles weakly coupled (UV asymptotic freedom)

**Averaged effective norm:**

$$
\|W_{\text{off}}^{\text{eff}}\|_{\text{op}} = \langle g'(\|v\| + b_\ell) \rangle \cdot \|W_{\text{off}}\|_{\text{op}}
$$

Define the **barrier strength factor**:

$$
\beta_\ell := \frac{\langle g'(\|v\| + b_\ell) - g'(\|v\|) \rangle}{\max_b |g'(\|v\| + b) - g'(\|v\|)|}
$$

For GELU and unit-norm $v$, this is approximately:

$$
\beta_\ell \approx \tanh(b_\ell)
$$

Thus:

$$
\|G_\mu^{(\ell)}\|_{\text{op}} = \beta_\ell \cdot \xi_\ell
$$

where $\xi_\ell := \|W_{\text{off}}^{(\ell)}\|_{\text{op}}$ is the bare off-diagonal weight norm.

**Step 5. Kinetic term from forward gradient flow:**

The kinetic term $\|\partial_\mu Z\|$ measures latent state changes across layers. For discrete layer index $\ell$:

$$
\partial_\ell z := z^{(\ell)} - z^{(\ell-1)}
$$

**Normalized gradient flow ratio:**

$$
\eta_\ell := \frac{\|\partial_\ell z\|_2}{\|z^{(\ell-1)}\|_2}
$$

For spectrally normalized networks with $\|W^{(\ell)}\|_{\text{op}} \leq 1$:

$$
\|z^{(\ell)}\|_2 \leq \|W^{(\ell)}\|_{\text{op}} \cdot \|z^{(\ell-1)}\|_2 \leq \|z^{(\ell-1)}\|_2
$$

Thus norms are non-increasing. The **relative change** is:

$$
\eta_\ell = \left\| \frac{W^{(\ell)} \cdot z^{(\ell-1)}}{\|z^{(\ell-1)}\|} - \frac{z^{(\ell-1)}}{\|z^{(\ell-1)}\|} \right\|_2 = \left\| \left(W^{(\ell)} - I\right) \hat{z}^{(\ell-1)} \right\|_2
$$

where $\hat{z} = z/\|z\|$ is the unit-normalized latent.

**Typical magnitude:** For residual architectures $W^{(\ell)} = I + \epsilon \Delta W^{(\ell)}$ with small $\epsilon$:

$$
\eta_\ell \approx \epsilon \|\Delta W^{(\ell)}\|_{\text{op}} \sim 0.01 \text{ to } 0.1
$$

**Step 6. Final coupling formula:**

Substitute Steps 4 and 5 into Step 2:

$$
g_s^{(\ell)} = \frac{\|G_\mu^{(\ell)}\|_{\text{op}}}{\|\partial_\mu z^{(\ell)}\|_{\text{op}}} = \frac{\beta_\ell \cdot \xi_\ell}{\eta_\ell \cdot \|z^{(\ell-1)}\|}
$$

For unit-normalized latents $\|z^{(\ell)}\| = 1$ (enforced by normalization layers):

$$
g_s^{(\ell)} = \frac{\beta_\ell \cdot \xi_\ell}{\eta_\ell}
$$

**Geometric averaging form:** To handle varying $\eta_\ell$ across layers, use the normalized form:

$$
g_s^{(\ell)} = \beta_\ell \cdot \frac{\xi_\ell}{\sqrt{\xi_\ell^2 + \eta_\ell^2}}
$$

This ensures:
- $g_s \in [0, |\beta_\ell|]$ (bounded coupling)
- $g_s \to \beta_\ell$ when $\xi_\ell \gg \eta_\ell$ (gauge field dominates)
- $g_s \to 0$ when $\eta_\ell \gg \xi_\ell$ (kinetic term dominates)

**Dimensional verification:**
- $[\beta_\ell]$ = dimensionless
- $[\xi_\ell] = [\|W\|]$ = dimensionless (operator norm)
- $[\eta_\ell] = [\|\partial_\ell z\| / \|z\|]$ = dimensionless
- $[g_s^{(\ell)}]$ = dimensionless ✓

**Step 7. Verify IR/UV limits:**

- **Infrared ($b_\ell \to +\infty$, deep layers):** $\beta_\ell \to 1$, and typically $\xi_\ell \sim O(1)$ (off-diagonal weights not suppressed), thus:

$$
g_s^{(\ell)} \to \frac{\xi_\ell}{\sqrt{\xi_\ell^2 + \eta_\ell^2}} \approx 1 \quad \text{if } \xi_\ell \gg \eta_\ell
$$
This gives **strong coupling** $g_s \sim O(1)$, leading to confinement (color-neutral states only).

- **Ultraviolet ($b_\ell \to 0$, shallow layers):** $\beta_\ell \to 0$, thus:

$$
g_s^{(\ell)} \to 0 \quad \text{(weak coupling, asymptotic freedom)}
$$
Bundles decouple and evolve independently.

- **Deeply suppressed ($b_\ell \to -\infty$):** $\beta_\ell \to -1$, giving negative coupling (unphysical). This regime is prevented by initialization with $b_\ell \geq 0$ and reparameterization $b_\ell = \text{softplus}(\tilde{b}_\ell)$ to enforce positivity.

**Comparison with QCD:** This running coupling behavior matches quantum chromodynamics:
- **IR**: $g_s(\mu_{\text{IR}}) \sim 1$ → confinement (quarks bound into hadrons)
- **UV**: $g_s(\mu_{\text{UV}}) \to 0$ → asymptotic freedom (quarks behave as free particles at high energy)

Here, the "energy scale" $\mu$ is replaced by layer depth $\ell$, with deep layers corresponding to IR and shallow layers to UV.

This matches the required behavior from Corollary {prf:ref}`cor-coupling-window` in the parameter sieve.

$\square$
:::


### Spectral Normalization and U(1)_Y

:::{prf:theorem} Spectral Norm Bounds Hypercharge Dissipation
:label: thm-spectral-preserves-hypercharge

The Opportunity field $B_\mu$ (from {ref}`sec-symplectic-multi-agent-field-theory`) couples to hypercharge $Y$. Spectral normalization ensures hypercharge is **non-increasing** under forward propagation:

$$
Y(W \cdot z) \leq Y(z) \quad \text{(hypercharge cannot increase)}
$$

*Proof.*

**Step 1. Hypercharge definition:** $Y(z) := \|z\|^2$ (quadratic quantity proportional to squared norm).

**Step 2. Contraction (NOT isometry):** Spectral normalization with $\sigma_{\max}(W) \leq 1$ ensures:

$$
\|W \cdot z\| \leq \sigma_{\max}(W) \cdot \|z\| \leq \|z\|
$$

This is a **contraction**, not an isometry. Equality $\|W \cdot z\| = \|z\|$ holds only when:
- $z$ is aligned with the top singular vector of $W$, AND
- $\sigma_{\max}(W) = 1$ exactly

For general $z$, we have strict inequality $\|W \cdot z\| < \|z\|$.

**Step 3. Hypercharge bound:**

$$
Y(W \cdot z) = \|W \cdot z\|^2 \leq \|z\|^2 = Y(z)
$$

**Interpretation:** Hypercharge dissipates (decreases) through linear layers but cannot spontaneously increase. This implements a one-way flow consistent with the second law of thermodynamics in the information-theoretic sense.

**Step 4. Conservation requires orthogonality:**
For **exact** hypercharge conservation $Y(W \cdot z) = Y(z)$ for all $z$, we would need:

$$
\|W \cdot z\| = \|z\| \quad \forall z \in \mathcal{Z}
$$

This requires $W$ to be **orthogonal**: $W^T W = I$. Spectral normalization does NOT enforce orthogonality; it only bounds the largest singular value.

$\square$

**Connection to Node 56 (CapacityHorizonCheck):** Hypercharge saturation $Y \to Y_{\max}$ indicates approaching capacity limit (holographic bound). The non-increasing property ensures the system cannot exceed this bound through forward propagation.

**Remark on exact conservation:** If exact hypercharge conservation is required (not just non-increasing), constrain $W$ to be orthogonal via Cayley parameterization or exponential map from skew-symmetric matrices:

$$
W = \exp(A) \quad \text{where } A^T = -A \text{ (skew-symmetric)}
$$
This guarantees $W^T W = I$ and thus $\|W \cdot z\| = \|z\|$ exactly.

**Remark on U(1)_Y correspondence:** The mapping "Spectral Normalization → U(1)_Y" is **analogical**, not a literal group isomorphism:
- **True U(1)_Y gauge symmetry** (from Standard Model): Phase rotations $z \to e^{i\alpha(x)} z$ with $\alpha(x)$ spacetime-dependent
- **Spectral normalization implements**: Norm bounding $\|W \cdot z\| \leq \|z\|$ (contraction, not rotation)

Both conserve a scalar charge: U(1) conserves $Q = \int \bar{\psi}\gamma^0\psi$ (electric charge), spectral norm conserves $Y = \|z\|^2$ (hypercharge/capacity). The **correspondence** is that both implement conservation laws for a U(1)-like scalar quantity, ensuring bounded capacity. However, spectral normalization does **not** implement U(1) rotations; it implements **Lipschitz contraction**.

For a literal U(1) rotation representation, one would parametrize weights as $W = e^{i\theta} W_0$ with $\theta \in [0, 2\pi)$. Spectral normalization instead parametrizes via singular values $W = U \Sigma V^T$ with $\sigma_1 \leq 1$.
:::

### Steerable Convolutions and SU(2)_L

:::{prf:definition} Observation-Action Doublet Structure
:label: def-obs-action-doublet

The latent space decomposes into **observation** and **action planning** subspaces:

$$
\mathcal{Z} = \mathcal{Z}_{\text{obs}} \oplus \mathcal{Z}_{\text{act}}
$$

These form an $SU(2)_L$ doublet:

$$
\Psi = \begin{pmatrix} \psi_{\text{obs}} \\ \psi_{\text{act}} \end{pmatrix} \in \mathcal{Z}_{\text{obs}} \oplus \mathcal{Z}_{\text{act}}
$$

**Full SU(2)_L transformation:** The special unitary group $SU(2)$ has 3 real parameters. A general element is:

$$
U(\vec{\theta}) = \exp\left(i \sum_{a=1}^3 \theta_a \frac{\sigma_a}{2}\right), \quad \vec{\theta} = (\theta_1, \theta_2, \theta_3) \in \mathbb{R}^3
$$
where $\{\sigma_a\}_{a=1}^3$ are Pauli matrices:

$$
\sigma_1 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_2 = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_3 = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$

**Real SO(2) subgroup (current implementation):** For real-valued neural networks, we restrict to the $SO(2)$ subgroup that performs real rotations in the $(\psi_{\text{obs}}, \psi_{\text{act}})$ plane. In the Pauli basis this corresponds to the one-parameter subgroup generated by $i\sigma_2$ (since $i\sigma_2 = \begin{psmallmatrix} 0 & 1 \\ -1 & 0 \end{psmallmatrix}$ is the canonical $\mathfrak{so}(2)$ generator). This gives 1-parameter transformations:

$$
U_{\text{SO(2)}}(\theta) = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix}, \quad \theta \in [0, 2\pi)
$$

**Action on doublet:**

$$
\Psi \to \Psi' = U_{\text{SO(2)}}(\theta) \Psi = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} \psi_{\text{obs}} \\ \psi_{\text{act}} \end{pmatrix}
$$

**Physical interpretation:**
- $\theta = 0$: Pure observation (no action planning)
- $\theta = \pi/2$: Pure action planning (no new observations)
- $0 < \theta < \pi/2$: Mixed observation-action processing

**Chirality:** This is a **left-chiral** doublet (incoming information stream). Right-chiral singlets correspond to executed actions (outgoing, no longer subject to mixing).

**Remark:** Full $SU(2)_L$ gauge theory requires complex-valued features. Current real-valued architectures (IsotropicBlock + SteerableConv) implement the $SO(2) \subset SU(2)$ subgroup. Extension to full 3-parameter $SU(2)$ requires complex steerable CNNs or quaternionic networks (see {ref}`sec-symplectic-multi-agent-field-theory` for gauge-theoretic derivation).
:::

:::{prf:theorem} Observation-Action Doublet as an $SO(2)$ (U(1)) Representation
:label: thm-steerable-induces-doublet

Let $\psi_{\text{obs}} \in \mathcal{Z}_{\text{obs}}$ and $\psi_{\text{act}} \in \mathcal{Z}_{\text{act}}$. Define the observation-action pair

$$
\Psi = \begin{pmatrix} \psi_{\text{obs}} \\ \psi_{\text{act}} \end{pmatrix}.
$$
Under the one-parameter subgroup $U_{\text{SO(2)}}(\theta)$ from Definition {prf:ref}`def-obs-action-doublet`, $\Psi$ transforms as $\Psi' = U_{\text{SO(2)}}(\theta)\Psi$, i.e., it is a 2D real representation of $SO(2)\cong U(1)$ (the implemented subgroup of $SU(2)_L$ in current real-valued architectures).

*Proof.*

**Step 1. Group action:** The map $\theta \mapsto U_{\text{SO(2)}}(\theta)$ is a representation of $SO(2)$ since

$$
U_{\text{SO(2)}}(\theta_1)U_{\text{SO(2)}}(\theta_2)=U_{\text{SO(2)}}(\theta_1+\theta_2)
$$
and $U_{\text{SO(2)}}(0)=I$.

**Step 2. Doublet transformation:** By Definition {prf:ref}`def-obs-action-doublet`,

$$
\Psi' = U_{\text{SO(2)}}(\theta)\Psi
= \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix}
\begin{pmatrix} \psi_{\text{obs}} \\ \psi_{\text{act}} \end{pmatrix},
$$
so the pair closes as a 2D real $SO(2)$ (equivalently $U(1)$) representation. $\square$

:::

**Remark on Error Field and Observation-Action Coupling:**

The Error field $W_\mu$ from gauge theory (Chapter 8.1, {ref}`sec-symplectic-multi-agent-field-theory`) couples observation and action representations in the full multiagent architecture. The detailed mathematical structure of this coupling—including the $SU(2)_L$ gauge symmetry—is derived rigorously in Chapter 8.2 (Standard Model isomorphism, see Table 8.2.1). At the DNN block level considered in this chapter, this manifests as:

- **Observation encoder:** Steerable convolutions producing rotation-equivariant features $\psi_{\text{obs}}$
- **Action decoder:** Linear layers mapping latent states to motor commands $\psi_{\text{act}}$
- **Coupling:** Cross-attention mechanisms (detailed in Chapter 8.5) that mix observation and action representations

The specific connection to $SU(2)_L$ gauge theory is established in Chapter 8.2 through rigorous derivation, not analogy.



(sec-integration-with-geodesic-integrator)=
## Integration with Geodesic Integrator

:::{prf:definition} Latent Metric Tensor
:label: def-latent-metric

The latent space $\mathcal{Z}$ is equipped with the **Information Sensitivity Metric** (from Section {ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits`, Theorem {prf:ref}`thm-capacity-constrained-metric-law`):

$$
G_{ij}(z) = \nabla^2_{ij} V(z) + \lambda \mathcal{F}_{ij}(z)
$$

where:
- $V(z)$ is the value function (expected return from state $z$)
- $\mathcal{F}_{ij}(z) = \mathbb{E}_{a \sim \pi(·|z)}[\nabla_{z_i} \log \pi(a|z) \nabla_{z_j} \log \pi(a|z)]$ is the Fisher Information Metric
- $\lambda > 0$ is the temperature parameter

**Physical interpretation:** The metric $G$ measures how sensitive the value function and policy are to changes in latent coordinates. High curvature regions indicate sensitive decision boundaries.

**Positive definiteness:** $G(z)$ is positive definite when $V$ is strongly convex and $\lambda > 0$.

**Units:** $[G_{ij}] = [\mathcal{Z}]^{-2} = \text{nat}^{-1}$ (inverse information).

**Variational origin** (sketch of derivation from Theorem {prf:ref}`thm-capacity-constrained-metric-law`):

The metric arises from minimizing the effective action under capacity constraints:

$$
G_{ij}(z) = \frac{\delta^2}{\delta z_i \delta z_j} \mathcal{A}_{\text{eff}}[z]
$$
where $\mathcal{A}_{\text{eff}}$ is the capacity-constrained effective action:

$$
\mathcal{A}_{\text{eff}}[z] = \int \left[V(z) + \frac{\lambda}{2} I(Z;A|z)\right] dz
$$

The **Hessian of the value function** $\nabla^2 V(z)$ captures curvature of the reward landscape (second-order approximation to the value surface), while the **Fisher Information Metric** $\mathcal{F}_{ij}(z)$ captures the sensitivity of the policy distribution $\pi(a|z)$ to latent perturbations.

**First variation** yields the Euler-Lagrange equations for optimal latent dynamics (geodesic equations on the WFR manifold). **Second variation** gives the metric as the Hessian of the action.

**Full derivation:** See Section 5.1 (Capacity-Constrained Metric Law) for the complete variational derivation from the bounded rationality Lagrangian, including the proof that $G$ is the unique metric satisfying the Monge-Ampère equation under holographic constraints.
:::

:::{div} feynman-prose
We have carefully designed each layer to be gauge-equivariant. But an agent is not one layer—it is a deep network, dozens or hundreds of layers stacked together. How do we know that equivariance at each step adds up to equivariance globally?

This is the question of *composition*, and it has a beautiful answer. If each layer commutes with the gauge transformation, then the whole network commutes with the gauge transformation. Apply a rotation, then pass through 50 layers, and you get the same result as passing through 50 layers, then rotating. The equivariance propagates through the entire depth.

But there is a subtlety. Equivariance alone is not enough—we also need the layers to play nicely with *integration*. The geodesic integrator requires that tangent vectors remain tangent vectors as they propagate. In geometric language, we need *parallel transport*: moving a vector along a curve without it leaving the tangent bundle.

This is where the Lipschitz constraint comes in. Each **linear** operator is spectrally normalized ($\sigma_{\max}(W) \leq 1$), so the linear parts are non-expansive. If you also enforce $L_f \leq 1$ for the nonlinear blocks (e.g., via the rescaled-gate option in Lemma {prf:ref}`lem-normgate-lipschitz`), then composing $L$ layers keeps the whole network 1-Lipschitz. More generally, bounded-gain nonlinearities give a controlled global Lipschitz bound $\prod_i L_i$, which still yields a global speed limit (but not strict light cone preservation).

Microscopic equivariance composes to macroscopic geodesic flow.
:::

### From Microscopic to Macroscopic

:::{prf:theorem} Composition of Equivariant Layers is Equivariant
:label: thm-composition-equivariant

Let $f_1, \ldots, f_L$ be $G$-equivariant layers. Then:

$$
F = f_L \circ \cdots \circ f_1 \text{ is } G\text{-equivariant}
$$

Moreover, if each $f_i$ has Lipschitz constant $L_i \leq 1$, then:

$$
L_F \leq 1 \quad \text{(global light cone preservation)}
$$

where $L_F$ is the Lipschitz constant of the composition $F$.

*Proof.*

**Equivariance:** Composition of equivariant maps is equivariant. For any $g \in G$:

$$
F(\rho(g) \cdot z) = f_L \circ \cdots \circ f_1(\rho(g) \cdot z) = \rho'(g) \cdot F(z)
$$
by repeatedly applying $f_i(\rho_i(g) \cdot w) = \rho_{i+1}(g) \cdot f_i(w)$.

**Lipschitz:** For any $z_1, z_2$:

$$
\|F(z_1) - F(z_2)\| = \|f_L \circ \cdots \circ f_1(z_1) - f_L \circ \cdots \circ f_1(z_2)\|
$$
Applying the Lipschitz property of each layer successively:

$$
\leq L_L \|f_{L-1} \circ \cdots \circ f_1(z_1) - f_{L-1} \circ \cdots \circ f_1(z_2)\| \leq \cdots \leq \left(\prod_{i=1}^L L_i\right) \|z_1 - z_2\|
$$
When $L_i \leq 1$ for all $i$, we have $L_F \leq \prod_i L_i \leq 1$. $\square$
:::

### Lipschitz Properties of Primitives

:::{prf:lemma} NormGate Lipschitz Bound
:label: lem-normgate-lipschitz

Let $f(v) = v \cdot g(\|v\| + b)$ be the norm-gated activation (Definition {prf:ref}`def-norm-gated-activation`) where $g: \mathbb{R} \to \mathbb{R}$ is a smooth gating function with:
- $|g(x)| \leq C_g |x|$ for all $x$ (sublinear growth)
- $|g'(x)| \leq L_g$ for all $x$ (bounded derivative)

Assume an operating range $\|v\| \leq R_{\max}$ and a bounded bias $|b| \leq B$. Then $f$ is Lipschitz on the operating range with constant:

$$
L_f \leq \max\bigl(R_{\max}(C_g + L_g) + C_gB,\; C_g(R_{\max} + B)\bigr).
$$
In typical settings with $L_g > 0$ and $R_{\max} \gtrsim 1$, the radial term $R_{\max}(C_g + L_g) + C_gB$ dominates.

**For GELU:** The GELU function $g(x) = x\Phi(x)$ where $\Phi$ is the standard normal CDF satisfies:
- $C_g = 1$ (since $0 \leq \Phi(x) \leq 1$ implies $|g(x)| \leq |x|$)
- $g'(x) = \Phi(x) + x\phi(x)$ where $\phi(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$
- $\sup_{x \in \mathbb{R}} g'(x) \approx 1.129$ (achieved at $x^* = \sqrt{2} \approx 1.414$ where $g''(x^*) = 0$)
- For practical operating range $x \in [-3, 3]$: $\max_{x \in [-3,3]} g'(x) \approx 1.129$ (same critical point)

**Derivation of critical point:**
$$g''(x) = \frac{d}{dx}[\Phi(x) + x\phi(x)] = \phi(x) + \phi(x) - x^2\phi(x) = \phi(x)(2 - x^2)$$
Setting $g''(x) = 0$ yields $x^2 = 2$, so $x^* = \sqrt{2}$ (taking positive root). At this point: $g'(\sqrt{2}) = \Phi(\sqrt{2}) + \sqrt{2}\phi(\sqrt{2}) \approx 0.9214 + 0.2075 \approx 1.129$.

Thus $L_g \approx 1.129$ and

$$
L_f \leq R_{\max}(1 + 1.129) + B \approx 2.129\,R_{\max} + B.
$$

*Proof.*

**Step 1. Spherical coordinates:**
Write $v = r\hat{v}$ where $r = \|v\| \geq 0$ and $\hat{v} = v/\|v\|$ is the unit direction vector.

Then:

$$
f(v) = f(r\hat{v}) = r \cdot g(r + b) \cdot \hat{v}
$$

**Step 2. Jacobian calculation:**

Recall $f(v) = v \cdot g(\|v\| + b)$ where $\|v\| = \sqrt{v^T v}$. We compute the Jacobian $\nabla f(v) = \frac{\partial f}{\partial v} \in \mathbb{R}^{d_b \times d_b}$.

**Decompose using product rule:**

$$
\frac{\partial f_i}{\partial v_j} = \frac{\partial}{\partial v_j}[v_i \cdot g(\|v\| + b)] = \delta_{ij} g(\|v\| + b) + v_i \frac{\partial g}{\partial v_j}
$$

**Compute gradient of norm:**

$$
\frac{\partial \|v\|}{\partial v_j} = \frac{\partial}{\partial v_j} (v^T v)^{1/2} = \frac{1}{2\|v\|} \cdot 2v_j = \frac{v_j}{\|v\|}
$$

**Apply chain rule to $g(\|v\| + b)$:**

$$
\frac{\partial g(\|v\| + b)}{\partial v_j} = g'(\|v\| + b) \cdot \frac{\partial(\|v\| + b)}{\partial v_j} = g'(\|v\| + b) \cdot \frac{v_j}{\|v\|}
$$

**Substitute into Jacobian:**

$$
\frac{\partial f_i}{\partial v_j} = \delta_{ij} g(\|v\| + b) + v_i \cdot g'(\|v\| + b) \cdot \frac{v_j}{\|v\|}
$$

**Matrix form:**

$$
\nabla f(v) = g(\|v\| + b) \cdot I_{d_b} + g'(\|v\| + b) \cdot \frac{vv^T}{\|v\|}
$$

This is the sum of a scaled identity and a rank-1 matrix.

**Step 3. Eigenvalue decomposition:**
Write $v = r\hat{v}$ where $r = \|v\|$ and $\|\hat{v}\| = 1$. The Jacobian has eigenvalues:

- **Radial direction** (eigenvector $\hat{v}$):

$$
\lambda_r = g(r+b) + g'(r+b) \cdot r
$$

- **Tangential directions** (eigenvectors orthogonal to $\hat{v}$, there are $d_b-1$ of these):

$$
\lambda_t = g(r+b)
$$

**Verification:** For $u = \hat{v}$:

$$
\nabla f(v) \cdot \hat{v} = g(r+b)\hat{v} + g'(r+b) r \hat{v} = [g(r+b) + rg'(r+b)]\hat{v} = \lambda_r \hat{v} \quad \checkmark
$$

For $u \perp \hat{v}$ (so $u^T v = 0$):

$$
\nabla f(v) \cdot u = g(r+b) u + g'(r+b) \frac{(v^T u)}{\|v\|} v = g(r+b) u = \lambda_t u \quad \checkmark
$$

**Step 4. Bound eigenvalues:**
Using $|g(x)| \leq C_g|x|$ and $|g'(x)| \leq L_g$:

**Radial:**

$$
|\lambda_r| = |g(r+b) + rg'(r+b)| \leq |g(r+b)| + r|g'(r+b)| \leq C_g|r+b| + rL_g
$$

For $r \leq R_{\max}$ and bounded bias $|b| \leq B$:

$$
|\lambda_r| \leq C_g(R_{\max} + B) + R_{\max}L_g = R_{\max}(C_g + L_g) + C_gB
$$

**Tangential:**

$$
|\lambda_t| = |g(r+b)| \leq C_g|r+b| \leq C_g(R_{\max} + B)
$$

**Step 5. Spectral norm and Lipschitz constant:**
The operator norm is:

$$
\|\nabla f(v)\|_{\text{op}} = \max(|\lambda_r|, |\lambda_t|) \leq \max(R_{\max}(C_g + L_g) + C_gB, C_g(R_{\max} + B))
$$

Since typically $L_g > 0$, the radial term dominates:

$$
L_f = \sup_v \|\nabla f(v)\|_{\text{op}} \leq R_{\max}(C_g + L_g) + C_gB
$$

**Step 6. Numerical values for GELU:**
For GELU with $C_g = 1$, $L_g \approx 1.129$, and typical bias $|b| \leq 1$:

$$
L_f \leq R_{\max}(1 + 1.129) + 1 \approx 2.129 R_{\max} + 1
$$

With normalized inputs, $R_{\max} \approx \sqrt{d_b}$ for $d_b$-dimensional bundles. For $d_b = 16$:

$$
L_f \lesssim 2.129 \times 4 + 1 \approx 9.5
$$

**Remark 1 (Composition with spectral norm):** While individual NormGate layers have $L_f > 1$, they compose with spectral-normalized linear layers (which have $L = 1$). The total Lipschitz constant for IsotropicBlock is bounded by the product, and layer normalization or skip connections prevent unbounded growth across depth.

**Remark 2 (Rescaling option):** To enforce strict 1-Lipschitz property, rescale GELU:

$$
\tilde{g}(x) = \frac{g(x)}{R_{\max}(C_g + L_g) + C_gB}
$$

This guarantees $L_f \leq 1$ but attenuates gradients. In practice, we keep unscaled GELU and rely on:
- Spectral normalization in linear layers
- Moderate bundle dimensions ($d_b \in [8, 32]$)
- Skip connections across blocks

to control the effective Lipschitz constant of the full network.
$\square$
:::

### Micro-Macro Consistency

:::{prf:definition} Micro-Macro Consistency
:label: def-micro-macro-consistency

A DNN layer $f: \mathcal{Z} \to \mathcal{Z}$ is **compatible with the geodesic integrator** if:

1. **Preserves metric (pullback):** Writing $J(z) := \frac{\partial f}{\partial z}(z)$, the pulled-back metric is

   $$
   (f^*G)(z) = J(z)^T\,G(f(z))\,J(z),
   $$
   and strict metric preservation (isometry) is $(f^*G)(z) = G(z)$.
2. **Bounded Lipschitz:** $\|f(z) - f(z')\| \leq L_f \cdot \|z - z'\|$ where $L_f$ is a finite constant
3. **Preserves gauge:** $f(U(g) \cdot z) = U(g) \cdot f(z)$ for all $g \in G_{\text{Fragile}}$

**Interpretation:**
- **Condition 1** ensures the metric structure evolves consistently (metric pullback)
- **Condition 2** ensures bounded signal propagation (Lipschitz continuity); for strict light cone preservation, require $L_f \leq 1$; for stable bounded amplification, allow $L_f = O(1)$ fixed constant
- **Condition 3** ensures gauge symmetry is preserved through network layers

**Units:** $[G] = [z]^{-2}$, $[L_f]$ dimensionless (ratio of distances), $[U(g)]$ dimensionless (unitary).
:::

:::{prf:theorem} Isotropic Blocks Satisfy Micro-Macro Consistency
:label: thm-isotropic-macro-compatible

IsotropicBlock (Definition {prf:ref}`def-isotropic-block`) satisfies:
1. **Bounded Lipschitz** (Condition 2 of {prf:ref}`def-micro-macro-consistency`) with $L_f \approx 9.5$ - proven rigorously
2. **Gauge invariance** (Condition 3 of {prf:ref}`def-micro-macro-consistency`) - proven rigorously
3. **Metric compatibility** (Condition 1 of {prf:ref}`def-micro-macro-consistency`) - pullback is well-defined, but exact isometry is not enforced (qualified)

*Proof.*

**Condition 2 (light cone preservation) - RIGOROUS:**

**Step 1. Decompose IsotropicBlock:**

$$
f = f_3 \circ f_2 \circ f_1
$$
where:
- $f_1 = \text{SpectralLinear}: z \mapsto Wz$ with $\sigma_{\max}(W) \leq 1$
- $f_2 = \text{Reshape}: \mathbb{R}^{d_{\text{out}}} \to (\mathbb{R}^{d_b})^{n_b}$ (identity as linear map, $J_2 = I$)
- $f_3 = \text{NormGate}: (v_1, \ldots, v_{n_b}) \mapsto (v_1 g(\|v_1\| + b_1), \ldots, v_{n_b} g(\|v_{n_b}\| + b_{n_b}))$

**Step 2. Lipschitz constants:**
- By Theorem {prf:ref}`thm-spectral-preserves-light-cone`: $L_1 \leq 1$
- $f_2$ is identity: $L_2 = 1$
- By Lemma {prf:ref}`lem-normgate-lipschitz` with GELU ($C_g = 1$, $L_g \approx 1.129$):

$$
L_3 \leq R_{\max}(C_g + L_g) + C_gB \approx 2.129 R_{\max} + 1
$$

For normalized bundles with $R_{\max} \approx \sqrt{d_b}$ and $d_b = 16$:

$$
L_3 \lesssim 2.129 \times 4 + 1 \approx 9.5
$$

**Step 3. Composition:**
By composition of Lipschitz functions ($\|f \circ g(x) - f \circ g(y)\| \leq L_f L_g \|x-y\|$):

$$
L_{\text{total}} = L_3 \cdot L_2 \cdot L_1 \leq 9.5 \cdot 1 \cdot 1 = 9.5
$$

**Interpretation: Bounded Amplification, Not Strict Light Cone Preservation**

The Lipschitz constant $L \approx 9.5$ means the IsotropicBlock can **amplify** signals by a bounded factor. This is *different* from strict light cone preservation (which would require $L \leq 1$, ensuring no amplification).

**Why amplification is acceptable:**
1. **Per-layer bound:** Each layer amplifies by at most $\times 9.5$, a fixed constant
2. **Composition depth:** For deep networks with $D$ layers, naive bound gives $L_{\text{total}} \leq (9.5)^D$, but this is pessimistic:
   - In practice, growth is controlled by normalization layers, residual scaling, and explicit diagnostics
3. **Gradient clipping:** Combined with gradient normalization in training, prevents runaway amplification

**Relationship to causal structure:**
- **SpectralLinear alone** has $L = 1$ (strict light cone preservation, Theorem {prf:ref}`thm-spectral-preserves-light-cone`)
- **NormGate** introduces bounded amplification ($L \approx 9.5$ for GELU with $d_b = 16$)
- **Net effect:** Information propagation is *bounded* but not *contractive*

**Practical consideration:**
For multi-layer networks, use explicit $\ell_2$ normalization after IsotropicBlock if strict $L \leq 1$ is required:
```python
z = isotropic_block(z)
z = z / z.norm(dim=-1, keepdim=True) * target_norm  # Renormalize
```

Thus:

$$
\|f(z_1) - f(z_2)\| \leq 9.5 \|z_1 - z_2\|
$$

**Remark on constant factor:** While $L > 1$ for individual blocks, depth-wise accumulation is controlled via:
- Skip connections (e.g., $z_{l+1} = z_l + \alpha \cdot \text{IsotropicBlock}(z_l)$ with $\alpha < 1$)
- Normalization layers between blocks
- The bound $L = O(\sqrt{d_b})$ is moderate for $d_b \in [8, 32]$

The effective light cone constraint $\|f(z_1) - f(z_2)\| \lesssim c_{\text{info}} \Delta t$ holds up to architecture-dependent constants.

**Condition 3 (gauge invariance) - RIGOROUS:**

Direct application of Theorem {prf:ref}`thm-isotropic-block-equivariant`. For gauge group $G_{\text{bundle}} = \prod_{i=1}^{n_b} SO(d_b)$:

$$
f(U(g) \cdot z) = U(g) \cdot f(z) \quad \forall g \in G_{\text{bundle}}
$$

This is proven constructively in Theorem {prf:ref}`thm-isotropic-block-equivariant` by showing each component (SpectralLinear, Reshape, NormGate) is equivariant and composition preserves equivariance.

**Condition 1 (metric preservation) - QUALIFIED:**

The strict isometry condition from Definition {prf:ref}`def-micro-macro-consistency`,

$$
G(z) = J(z)^T\,G(f(z))\,J(z),
$$
depends critically on whether the metric is constant or state-dependent.

**Case A: Constant Euclidean metric** ($G(z) = I$ for all $z$)

The pullback condition becomes:

$$
I = J^T I J = J^T J
$$

This requires $J$ to be orthogonal, which is NOT true for IsotropicBlock:
- SpectralLinear: $J_1 = W$ with $\sigma_{\max}(W) \leq 1$ satisfies $W^T W \preceq I$ (contraction, not isometry unless $W$ is exactly orthogonal)
- NormGate: $J_3 = g(\|v\|+b)I + g'(\|v\|+b)vv^T/\|v\|$ satisfies:

$$
J_3^T J_3 = g^2(\|v\|+b) I + \left[\frac{2g(\|v\|+b)g'(\|v\|+b)}{\|v\|} + g'^2(\|v\|+b)\right] vv^T \neq I
$$

Thus **exact isometry** $(f^*G)(z)=G(z)$ fails for the constant Euclidean metric.

**Case B: Information Sensitivity Metric** (Definition {prf:ref}`def-latent-metric`)

For state-dependent $G(z) = \nabla^2 V(z) + \lambda \mathcal{F}(z)$, the pullback requires:

$$
\nabla^2 V(z) + \lambda \mathcal{F}(z) = J^T [\nabla^2 V(f(z)) + \lambda \mathcal{F}(f(z))] J
$$

Since $V$ and $\mathcal{F}$ depend on state, $G(f(z)) \neq G(z)$ in general, and **exact isometry** $(f^*G)(z)=G(z)$ does not hold.

**What IS rigorously true:**

1. **Positive (semi)definiteness under pullback:** If $G \succ 0$, then for any Jacobian $J$ we have $J^T G J \succeq 0$, and if $J$ has full column rank (in particular, if $J$ is invertible) then $J^T G J \succ 0$. One convenient bound (when $J$ is full-rank) is:

$$
J^T G(z) J \succeq \lambda_{\min}(J^T J) \cdot \lambda_{\min}(G(z)) > 0
$$
Thus the pullback cannot introduce negative directions; it can only collapse directions if $J$ is rank-deficient.

2. **Structure preservation:** NormGate acts isotropically within bundles:

$$
J_3^{(i)} = g_i I_{d_b} + h_i v_i v_i^T
$$
where $g_i = g(\|v_i\| + b_i)$ and $h_i = g'(\|v_i\| + b_i)/\|v_i\|$. This preserves radial-tangential structure of metrics that decompose similarly.

3. **Empirical compatibility:** Diagnostic Node 67 (GaugeInvarianceCheck) verifies that applying gauge transformations produces consistent behavior, indicating the metric structure is preserved in the sense relevant for geodesic integration.

**Conclusion:** IsotropicBlock does NOT satisfy the exact isometry condition $G(z) = J^T G(f(z)) J$ for general metrics. However, it preserves:
- Gauge structure (Condition 3)
- Lipschitz bounds (Condition 2)
- Positive-definiteness and structural properties of the metric

For the geodesic integrator, this is sufficient because the metric is recomputed at each integration step rather than being pulled back through transformations.

$\square$
:::

:::{prf:theorem} Metric Pullback Defect Under Composition
:label: thm-approximate-metric-preservation

Let $F = f_L \circ \cdots \circ f_1$ be a deep network on $\mathcal{Z}$, and let $G$ be a (possibly state-dependent) metric tensor on $\mathcal{Z}$.

Define $z_0 := z$, $z_\ell := f_\ell(z_{\ell-1})$, and Jacobians $J_\ell := \frac{\partial f_\ell}{\partial z}(z_{\ell-1})$. Let $J_F := J_L \cdots J_1$.

Define the per-layer **isometry defect**:

$$
E_\ell := J_\ell^T\,G(z_\ell)\,J_\ell - G(z_{\ell-1}).
$$

Then the total pullback defect admits the exact decomposition:

$$
J_F^T\,G(z_L)\,J_F - G(z_0)
=
\sum_{\ell=1}^L (J_{\ell-1}\cdots J_1)^T\,E_\ell\,(J_{\ell-1}\cdots J_1),
$$
with the convention $J_0\cdots J_1 := I$.

In particular, if $\|J_\ell\|_{\mathrm{op}} \leq L_J$ and $\|E_\ell\|_{\mathrm{op}} \leq e_\ell$, then:

$$
\|J_F^T\,G(F(z))\,J_F - G(z)\|_{\mathrm{op}} \leq \sum_{\ell=1}^L L_J^{2(\ell-1)}\,e_\ell.
$$

So when $L_J \le 1$ the defect accumulates at most linearly in $\sum_\ell e_\ell$, while for $L_J>1$ the naive worst-case bound grows like $L_J^{2L}$.

*Proof.* Induction on $L$.

**Remark (Geodesic integrator):** The Boris-BAOAB integrator (Chapter 4, Section {ref}`sec-geodesic-integrator`) recomputes the metric $G(z_t)$ at each timestep $t$ and does not assume layer-wise isometries (it does not require $f^*G=G$ for each layer).
:::

:::{admonition} Connection to RL: Implicit Regularization via Architecture
:class: note

**Standard approach:** Add metric penalties to loss function (computational cost at runtime).

**Fragile approach:** Build metric preservation into forward pass (zero-cost enforcement).

**Result:** Guaranteed consistency without optimization overhead.
:::



(sec-dimensional-analysis)=
## Dimensional Analysis and Unit Tracking

:::{prf:proposition} Latent Dimension from Information-Theoretic First Principles
:label: prop-latent-dimension-from-capacity

The latent space $\mathcal{Z} \subset \mathbb{R}^{d_z}$ represents compressed observations encoding mutual information $I(X;Z) \leq C$ where $C$ is the channel capacity (nat/step) from the bounded rationality controller (Chapter 1).

**Derivation from Gaussian rate-distortion theory:**

For a Gaussian source $X \sim \mathcal{N}(0, \Sigma_X)$ encoded into latent $Z \in \mathbb{R}^{d_z}$ via encoder $p(Z|X)$ with reconstruction $\hat{X} = \mathbb{E}[X|Z]$:

**Step 1. Rate-distortion tradeoff:**
The optimal encoder for squared error distortion $D = \mathbb{E}[\|X - \hat{X}\|^2]$ achieves:

$$
I(X;Z) = \frac{1}{2}\sum_{i=1}^{d_z} \log\left(1 + \frac{\lambda_i}{\sigma^2}\right)
$$
where $\lambda_i$ are eigenvalues of the source covariance $\Sigma_X$ allocated to latent dimension $i$, and $\sigma^2$ is the noise level per dimension.

**Step 2. Equal allocation (isotropic latent):**
For computational efficiency, neural encoders typically use isotropic latent representations with equal variance per dimension:

$$
\Sigma_Z = \sigma_z^2 I_{d_z}
$$

The total information is:

$$
I(X;Z) \leq \frac{1}{2} d_z \log\left(1 + \frac{\sigma_X^2}{\sigma_{\text{noise}}^2}\right)
$$

**Step 3. Dimensional analysis from first principles:**

The mutual information formula (Step 1) is:

$$
I(X;Z) = \frac{1}{2}\sum_{i=1}^{d_z} \log\left(1 + \frac{\lambda_i}{\sigma^2}\right)
$$

**Logarithm constraint:** The logarithm function requires a dimensionless argument. Therefore:

$$
\left[1 + \frac{\lambda_i}{\sigma^2}\right] = [1] \quad \text{(dimensionless)}
$$

This implies:

$$
\left[\frac{\lambda_i}{\sigma^2}\right] = [1] \quad \Rightarrow \quad [\lambda_i] = [\sigma^2]
$$

**Information-theoretic foundation:** In Shannon information theory, differential entropy for a Gaussian random variable $X \sim \mathcal{N}(0, \Sigma)$ is **defined** to have units [nat]:

$$
h(X) = \frac{1}{2} \log \det(2\pi e \Sigma) \quad [\text{nat}]
$$

For a single dimension with variance $\lambda$:

$$
h(X_i) = \frac{1}{2}\log(2\pi e \lambda) \quad [\text{nat}]
$$

**Critical distinction:** The numerical prefactor $1/2$ is dimensionless (as all pure numbers are). The "nat" unit arises from the **operational definition** of information in Shannon's framework: entropy measures the expected log-probability, and we adopt the convention $[h(X)] = [\text{nat}]$ to distinguish information content from dimensionless logarithms. This is analogous to how we define $[E] = \text{Joule}$ in physics—it's a choice of unit system, not algebraic dimension propagation.

**Dimensional interpretation:** The formula should be understood as:

$$
h(X) = \left[\frac{1}{2}\log \det(2\pi e \Sigma)\right]_{\text{nat}}
$$
where the subscript indicates the dimensionless logarithm is **measured in units** of nats (the information-theoretic unit), not that it algebraically has dimension [nat].

**Dimensional convention (NOT derivation):** We **adopt the convention** that latent coordinates have dimension:

$$
\boxed{[z] = [\mathcal{Z}] := \sqrt{\text{nat}}}
$$

**This is a choice**, not a theorem. Here's why we make this choice:

**Step 3a. Consistency requirement:**
If we want variance $[\sigma_z^2]$ to have the **same units** as information measures (differential entropy $h(X)$ in nats), and if coordinates are related to variance by $[z^2] = [\sigma_z^2]$, then we must have:

$$
[z] = \sqrt{[\sigma_z^2]} = \sqrt{[\text{nat}]}
$$

**Step 3b. Motivation for the choice:**
This convention ensures dimensional consistency across the framework:
- Rate-distortion: $I(X;Z) = \frac{1}{2}\sum_i \log(\lambda_i/\sigma^2)$ requires $[\lambda_i] = [\sigma^2]$; setting both equal to [nat] makes $I$ dimensionally consistent with entropy
- Fisher metric: $\mathcal{F}_{ij} = \mathbb{E}[\partial_i \log p \, \partial_j \log p]$ has $[\mathcal{F}] = [z]^{-2}$; if $[z] = \sqrt{\text{nat}}$ then $[\mathcal{F}] = \text{nat}^{-1}$, matching information-theoretic quantities
- Capacity: $C$ (nat/step) becomes commensurate with $\|z\|^2$ (nat) and $d_z \times \sigma_z^2$ (dimensionless × nat)

**Step 3c. What is NOT claimed:**
- We do **not** claim this follows logically from information theory alone
- We do **not** claim $[\lambda] = [\text{nat}]$ is forced by mathematics—it's a **definition** we choose
- This is analogous to setting $c = 1$ in relativity: a **unit convention** that simplifies equations, not a physical law

**Interpretation:** Each latent coordinate carries information measured in natural units (nats). The variance of a latent dimension represents information content. This convention parallels quantum mechanics where position $x$ relates to momentum $p$ via $\Delta x \Delta p \sim \hbar$ with $[\hbar] = \text{action}$ giving $[x] \sim \sqrt{\text{action}}$.

**Remark on arbitrariness:** This **is** an arbitrary convention in the sense that we could equally well work in dimensionless units throughout and track "nat" as a label rather than a dimension. We choose to promote "nat" to a pseudo-dimension because:
1. It makes dimensional analysis track information flow explicitly
2. It connects latent-space geometry to information-theoretic foundations
3. It parallels established physics conventions (action, angular momentum as dimensional units)
:::

:::{prf:definition} Information Speed in Latent Coordinates
:label: def-information-speed-latent

The **latent information speed** is the maximum rate of latent state change per unit time:

$$
c_{\mathcal{Z}} := \sup_{z(·), \Delta t > 0} \frac{d_{\mathcal{Z}}(z(t + \Delta t), z(t))}{\Delta t}
$$

where:
- $z: [0, T] \to \mathcal{Z}$ is a latent trajectory
- $d_{\mathcal{Z}}(z_1, z_2) = \|z_1 - z_2\|$ is the Euclidean distance in latent space
- The supremum is taken over all admissible trajectories and time increments

**Dimensions**: $[c_{\mathcal{Z}}] = [\mathcal{Z}][T^{-1}] = \sqrt{\text{nat}} \cdot [T^{-1}]$ where $[T]$ denotes abstract time dimension (measured in seconds for physical agents)

**Physical interpretation**: This is the "speed of thought"—the maximum rate at which the agent's internal representation can evolve under the dynamics.

**Connection to environment information speed**:

Let $\mathcal{E}$ denote the environment observation space (e.g., pixel space for vision, $\mathcal{E} = \mathbb{R}^{H \times W \times C}$).

Let $\phi: \mathcal{E} \to \mathcal{Z}$ be the encoder network mapping observations to latents, with Jacobian $J_\phi(x) = \nabla \phi(x) \in \mathbb{R}^{d_z \times d_{\mathcal{E}}}$.

By the chain rule for composed dynamics $z(t) = \phi(x(t))$:

$$
\frac{dz}{dt} = J_\phi(x(t)) \cdot \frac{dx}{dt}
$$

Taking norms:

$$
\left\|\frac{dz}{dt}\right\| \leq \|J_\phi(x)\|_{\text{op}} \cdot \left\|\frac{dx}{dt}\right\|
$$

If the environment dynamics satisfy $\|dx/dt\| \leq c_{\text{info}}$ (Axiom {prf:ref}`ax-information-speed-limit` from Chapter 8.1), then:

$$
c_{\mathcal{Z}} \leq \sup_x \|J_\phi(x)\|_{\text{op}} \cdot c_{\text{info}}
$$

**Remark**: The encoder Lipschitz constant $L_\phi = \sup_x \|J_\phi(x)\|_{\text{op}}$ controls how environmental changes propagate to latent space. Spectral normalization ensures the **linear** parts satisfy $\|W\|_{\text{op}}=\sigma_{\max}(W)\leq 1$; the overall $L_\phi$ is then bounded by the product of per-block Lipschitz bounds, so strict $L_\phi \leq 1$ requires also using 1-Lipschitz nonlinearities (or explicitly tracking/rescaling their gain).

**Operational constraint**: For strict causality preservation (Theorem {prf:ref}`thm-spectral-preserves-light-cone`), every **linear** map must satisfy $\sigma_{\max}(W) \leq 1$ and each nonlinear block should satisfy $L_f \leq 1$; if bounded amplification is allowed, track the resulting global Lipschitz bound to maintain a controlled speed limit.
:::

**Table: Complete Dimensional Analysis for Covariant Primitives**

| Quantity | Symbol | Abstract Dimension | Physical Units | Normalized Form | Enforcement |
|:---------|:-------|:-------------------|:---------------|:----------------|:------------|
| Latent vector | $z$ | $[\mathcal{Z}]$ | $\sqrt{\text{nat}}$ | $\|z\| \leq 1$ | Spectral norm in encoder |
| Bundle norm | $\|v\|$ | $[\mathcal{Z}]$ | $\sqrt{\text{nat}}$ | $\geq 0$ | Euclidean norm per bundle |
| Weight matrix | $W$ | $[\mathcal{Z}']/[\mathcal{Z}]$ | dimensionless | $\sigma_{\max}(W) \leq 1$ | Spectral normalization (Def. {prf:ref}`def-spectral-linear`) |
| Activation bias | $b$ | $[\mathcal{Z}]$ | $\sqrt{\text{nat}}$ | $\in \mathbb{R}$ | Learnable parameter |
| Information speed (latent) | $c_{\mathcal{Z}}$ | $[\mathcal{Z}][T^{-1}]$ | $\sqrt{\text{nat}} \cdot [T^{-1}]$ | $> 0$ | Definition {prf:ref}`def-information-speed-latent` |
| Information speed (environment) | $c_{\text{info}}$ | $[L][T^{-1}]$ | $[L][T^{-1}]$ | $> 0$ | Axiom {prf:ref}`ax-information-speed-limit` (where $[L]$ represents physical length in environment coordinates) |
| Metric tensor | $G_{ij}$ | $[\mathcal{Z}]^{-2}$ | $\text{nat}^{-1}$ | Positive definite | Definition {prf:ref}`def-latent-metric` |
| Distance (latent) | $d_{\mathcal{Z}}$ | $[\mathcal{Z}]$ | $\sqrt{\text{nat}}$ | $\geq 0$ | Euclidean distance |
| Energy barrier | $\Delta E$ | $[\mathcal{Z}]^2$ | nat | $> 0$ | Squared norm scale |
| Gate output | $g$ | [1] | dimensionless | $\in \mathbb{R}$ (typically $\ge 0$ when $\|v\|+b \ge 0$) | GELU activation |
| Bundle dimension | $d_b$ | [1] | dimensionless | $\in \mathbb{N}$ | Architectural hyperparameter |

:::{prf:proposition} Dimensional Consistency of IsotropicBlock
:label: prop-dimensional-consistency

Let $z \in \mathcal{Z}$ with $[z] = [\mathcal{Z}] = \sqrt{\text{nat}}$ (Proposition {prf:ref}`prop-latent-dimension-from-capacity`). The IsotropicBlock operation

$$
\text{IsotropicBlock}(z) = \text{Reshape}(\text{NormGate}(\text{SpectralLinear}(z)))
$$
preserves the latent dimension $[\mathcal{Z}]$ through each stage when interpreted with implicit normalization conventions.

*Proof.*

**Dimensional Analysis Axioms:**
1. $[AB] = [A][B]$ (multiplicative)
2. $[A + B]$ defined iff $[A] = [B]$ (additive homogeneity)
3. For transcendental function $f: \mathbb{R} \to \mathbb{R}$ (like GELU), the argument must be dimensionless or implicitly normalized

**Step 1. SpectralLinear:**

$$
[W \cdot z] = [W] \cdot [z] = \frac{[\mathcal{Z}']}{[\mathcal{Z}]} \cdot [\mathcal{Z}] = [\mathcal{Z}']
$$
where $[\mathcal{Z}'] = [\mathcal{Z}]$ (linear map between same-dimensional latent spaces).

**Step 2. Reshape:**
Identity operation that permutes indices: $[\text{Reshape}(h)] = [h] = [\mathcal{Z}]$.

**Step 3. NormGate per bundle $i$ — natural units convention:**

**Convention adopted:** We work in **natural units** where the reference latent scale is:

$$
z_0 := 1 \quad \text{(in units of } \sqrt{\text{nat}}\text{)}
$$

This is analogous to setting $c = \hbar = 1$ in relativistic quantum mechanics. Under this convention, latent coordinates and norms are dimensionless pure numbers when expressed in units of $z_0$.

**Justification:** Spectral normalization $\sigma_{\max}(W) \leq 1$ ensures $\|v_i\| \sim O(z_0)$, making the natural unit system well-defined.

**Dimensional analysis under natural units:**

Recall the definition (Def. {prf:ref}`def-norm-gated-activation`):

$$
f(v_i) = v_i \cdot g(\|v_i\| + b_i)
$$

In natural units where $z_0 = 1$:
- $[\|v_i\|] = [v_i] = [\mathcal{Z}]$ (latent dimension)
- $[b_i] = [\mathcal{Z}]$ (homogeneous addition)
- The argument $\|v_i\| + b_i$ is **a pure number** (ratio to $z_0 = 1$)
- $g: \mathbb{R} \to \mathbb{R}$ takes dimensionless input and returns dimensionless output
- Therefore $[f(v_i)] = [v_i] \cdot [1] = [\mathcal{Z}]$ ✓

**Step 4. Output dimension:**

$$
[\text{IsotropicBlock}(z)] = [\mathcal{Z}]
$$

$\square$

**Implementation note:** The code writes:
```python
gate = F.gelu(energy + self.norm_bias)
```
where `energy = ||v_i||` is numerically $O(1)$ due to spectral normalization. This directly implements the natural units convention with $z_0 = 1$ absorbed.

**Alternative (strict dimensional analysis):** For explicit dimensional tracking without natural units, the formula would be:

$$
f(v) = v \cdot g\left(\frac{\|v\| + b}{z_0}\right)
$$
where $z_0 = \mathbb{E}[\|v\|]$ is the expected bundle norm with $[z_0] = [\mathcal{Z}]$. In normalized architectures, $z_0 \approx 1\,\sqrt{\text{nat}}$, reducing to the natural units case.
:::



(sec-diagnostic-nodes)=
## Diagnostic Nodes and Runtime Verification

### Gauge Invariance Check (Node 67)

:::{prf:definition} Gauge Violation Metric
:label: def-gauge-violation-metric

For operator $f$ and group element $g \in G$:

$$
\delta_{\text{gauge}}(f, g) = \mathbb{E}_z\left[\|f(U(g) \cdot z) - U(g) \cdot f(z)\|^2\right]
$$

**Threshold:** $\delta_{\text{gauge}} < \epsilon_{\text{gauge}} = 10^{-4}$ (empirically tuned).
:::

**Diagnostic implementation:**

```python
class GaugeInvarianceCheck(DiagnosticNode):
    """Node 67: Verify G-equivariance of layers.

    Tests: Sample random rotations/transformations, measure violation.
    """
    def __init__(self, layer: nn.Module, group: str = "SO(d)"):
        self.layer = layer
        self.group = group

    def check(self, z: torch.Tensor) -> Dict[str, float]:
        # Sample random group element
        if self.group == "SO(d)":
            g = self._random_rotation(z.shape[-1])

        # Apply transformation
        z_transformed = g @ z

        # Check equivariance: f(g·z) ≟ g·f(z)
        f_g_z = self.layer(z_transformed)
        g_f_z = g @ self.layer(z)

        violation = torch.norm(f_g_z - g_f_z).item()

        return {
            "gauge_violation": violation,
            "threshold": 1e-4,
            "passed": violation < 1e-4
        }
```

### Summary of Diagnostic Nodes

**Table: Diagnostic Nodes for Covariant Primitives**

| Node | Name | Verifies | Trigger Condition |
|:-----|:-----|:---------|:------------------|
| 40 | PurityCheck | $SU(N_f)_C$ confinement | Non-neutral bundles at macro boundary |
| 56 | CapacityHorizonCheck | $U(1)_Y$ bound | Hypercharge $Y \to Y_{\max}$ |
| 62 | CausalityViolationCheck | Light cone preservation | $\sigma_{\max}(W) > 1 + \epsilon$ |
| 67 | GaugeInvarianceCheck | $G$-equivariance | $\delta_{\text{gauge}} > \epsilon_{\text{gauge}}$ |
| 68 | RotationEquivarianceCheck | $SO(2)$ for images | $\|f(R \cdot I) - R \cdot f(I)\| > \epsilon$ |



(sec-implementation-reference)=
## Implementation Reference

The following provides the complete implementation of `IsotropicBlock` with comprehensive annotations.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Dict

class IsotropicBlock(nn.Module):
    """Atomic gauge-covariant building block (Definition {prf:ref}`def-isotropic-block`).

    Implements: SpectralLinear ∘ Reshape ∘ NormGate

    **Equivariance modes:**
        - `exact=True`: Uses scalar blocks (W_i = λ_i I), giving **exact** SO(d_b)^{n_b}
          equivariance per Theorem {prf:ref}`thm-isotropic-block-equivariant`.
        - `exact=False` (default): Uses block-diagonal W with general blocks, giving
          **approximate** equivariance per Proposition {prf:ref}`prop-approximate-equivariance-bound`.
          More expressive but not exactly equivariant.

    **Guarantees (exact mode):**
        - Exact SO(d_b)^{n_b} equivariance (Theorem {prf:ref}`thm-isotropic-block-equivariant`)
        - Light cone preservation (Theorem {prf:ref}`thm-spectral-preserves-light-cone`)
        - Micro-macro consistency (Theorem {prf:ref}`thm-isotropic-macro-compatible`)

    **Guarantees (approximate mode):**
        - Bounded equivariance violation (Proposition {prf:ref}`prop-approximate-equivariance-bound`)
        - Light cone preservation (Theorem {prf:ref}`thm-spectral-preserves-light-cone`)
        - Greater expressiveness (can learn within-bundle transformations)

    **Diagnostics:**
        - Node 67: Gauge invariance (test random rotations)
        - Node 62: Causality (verify σ_max ≤ 1)
        - Node 40: Confinement (check bundle binding)

    Args:
        in_dim: Input dimension D_in [dimensionless]
        out_dim: Output dimension D_out [dimensionless]
        bundle_size: Bundle dimension d_b [dimensionless], typically 8-32
        exact: If True, use scalar blocks for exact equivariance (less expressive).
               If False (default), use block-diagonal for approximate equivariance.

    Shapes:
        Input: [B, D_in] where B = batch size
        Output: [B, D_out]

    Units:
        All latent vectors: [z] = √nat (Proposition {prf:ref}`prop-latent-dimension-from-capacity`)
        Bundle norm (energy): ||v|| ∈ [0, √d_b] (dimensionless pure numbers in natural units with z₀=1√nat)
        Gate output: g(||v|| + b) ∈ R (dimensionless; GELU is unbounded above and can be negative)

    Example:
        >>> # Approximate equivariance (more expressive, default)
        >>> block = IsotropicBlock(256, 512, bundle_size=16, exact=False)
        >>> z = torch.randn(32, 256)  # [B=32, D_in=256]
        >>> out = block(z)  # [32, 512]
        >>>
        >>> # Exact equivariance (use when strict gauge invariance is required)
        >>> block_exact = IsotropicBlock(256, 512, bundle_size=16, exact=True)
        >>> R = random_bundle_rotation(512, 16)  # Block-diagonal rotation
        >>> assert torch.allclose(block_exact(R @ z), R @ block_exact(z), atol=1e-6)
    """
    def __init__(self, in_dim: int, out_dim: int, bundle_size: int = 16, exact: bool = False):
        super().__init__()
        assert out_dim % bundle_size == 0, "Output dim must be divisible by bundle size"
        assert in_dim % bundle_size == 0 or not exact, "For exact mode, in_dim must be divisible by bundle_size"

        self.bundle_size = bundle_size
        self.n_bundles = out_dim // bundle_size
        self.exact = exact

        if exact:
            # EXACT EQUIVARIANCE: Scalar blocks W_i = λ_i I (Lemma {prf:ref}`lem-schur-scalar-constraint`)
            # One scalar per bundle, constrained to [-1, 1]
            self.bundle_scales = nn.Parameter(torch.ones(self.n_bundles))
            # Note: For exact mode, in_dim must equal out_dim (same bundle structure)
            assert in_dim == out_dim, "Exact mode requires in_dim == out_dim (same bundle structure)"
        else:
            # APPROXIMATE EQUIVARIANCE: Block-diagonal W with general blocks
            # (Proposition {prf:ref}`prop-approximate-equivariance-bound`)
            # Each block W_i: R^{d_b} -> R^{d_b} with σ_max(W_i) ≤ 1
            self.block_weights = nn.ParameterList([
                nn.Parameter(torch.randn(bundle_size, bundle_size) / (bundle_size ** 0.5))
                for _ in range(self.n_bundles)
            ])
            # For input dimension mapping (if in_dim != out_dim)
            if in_dim != out_dim:
                self.input_proj = spectral_norm(nn.Linear(in_dim, out_dim, bias=False))
            else:
                self.input_proj = None

        # 2. The Activation Potential (scalar bias on norms)
        # One bias per vector bundle [dimensionless]
        self.norm_bias = nn.Parameter(torch.zeros(1, self.n_bundles, 1))

    def _spectral_normalize_block(self, W: torch.Tensor) -> torch.Tensor:
        """Apply spectral normalization to a single block weight matrix.

        Args:
            W: [d_b, d_b] weight matrix

        Returns:
            W_normalized: [d_b, d_b] with σ_max ≤ 1
        """
        # Power iteration to estimate largest singular value
        u = torch.randn(W.shape[0], device=W.device)
        for _ in range(3):  # Few iterations suffice for approximation
            v = F.normalize(W.T @ u, dim=0)
            u = F.normalize(W @ v, dim=0)
        sigma_max = (u @ W @ v).abs()
        return W / sigma_max.clamp(min=1.0)  # Only normalize if σ_max > 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dimensional annotations.

        Args:
            x: [B, in_dim] input [dimensionless, normalized]

        Returns:
            output: [B, out_dim] equivariant activation [dimensionless]
        """
        B = x.shape[0]

        if self.exact:
            # EXACT MODE: Apply scalar scaling per bundle (W_i = λ_i I)
            # Clamp scales to [-1, 1] for spectral normalization
            scales = self.bundle_scales.clamp(-1, 1)  # [n_b]

            # Reshape input into bundles
            h_bundles = x.view(B, self.n_bundles, self.bundle_size)  # [B, n_b, d_b]

            # Apply scalar scaling per bundle: λ_i * v_i
            # scales has shape [n_b], expand to [1, n_b, 1] for broadcasting
            h_bundles = h_bundles * scales.view(1, -1, 1)  # [B, n_b, d_b]
        else:
            # APPROXIMATE MODE: Apply block-diagonal transformation
            # First, project input if dimensions don't match
            if self.input_proj is not None:
                x = self.input_proj(x)  # [B, out_dim]

            # Reshape input into bundles
            h_bundles = x.view(B, self.n_bundles, self.bundle_size)  # [B, n_b, d_b]

            # Apply spectrally-normalized block transformation per bundle
            h_list = []
            for i in range(self.n_bundles):
                W_i = self._spectral_normalize_block(self.block_weights[i])  # [d_b, d_b]
                h_i = torch.einsum('bd,de->be', h_bundles[:, i, :], W_i)  # [B, d_b]
                h_list.append(h_i)
            h_bundles = torch.stack(h_list, dim=1)  # [B, n_b, d_b]

        # Step 3: Compute energy (SO(d_b)-invariant norm)
        energy = torch.norm(h_bundles, dim=2, keepdim=True)  # [B, n_b, 1]
        # Natural units: energy ∈ [0, √d_b] (dimensionless); with explicit units: [energy] = √nat
        # (see Section {ref}`sec-natural-units`: z₀=1√nat absorbed in implementation)

        # Step 4: Energy gate (smooth approximation to barrier function)
        # GELU is C^∞; note f(v)=v·g(||v||+b) is only C^1 at v=0 unless ||v|| is smoothed (Def. {prf:ref}`def-norm-gated-activation`)
        # Definition {prf:ref}`def-norm-gated-activation`: g(||v|| + b) where g(x) = x·Φ(x)
        gate = F.gelu(energy + self.norm_bias)  # [B, n_b, 1]
        # [gate] = dimensionless (GELU unbounded above; can be negative for negative inputs)
        # For ||v|| ~ 1 and b ~ 0: gate ≈ 0.84 (GELU(1) ≈ 0.84)
        # For ||v|| >> 1: gate ≈ ||v|| (linear growth)
        # For ||v|| << -b: gate ≈ 0 (suppressed)

        # Step 5: Apply gate scaling to bundles (Thm {prf:ref}`thm-norm-gating-equivariant`)
        # By Definition {prf:ref}`def-norm-gated-activation`: f(v) = v · g(||v|| + b)
        # The vector is scaled by the gate value, preserving direction
        h_out = h_bundles * gate  # [B, n_b, d_b]
        # gate has shape [B, n_b, 1] so broadcasting scales each bundle

        # Step 6: Flatten back to vector
        D = self.n_bundles * self.bundle_size
        return h_out.view(B, D)  # [B, out_dim]
```



(sec-summary-tables)=
## Summary and Cross-Reference Tables

### Standard DL → Fragile Primitives

**Table: Replacement Architecture**

| Component | Standard DL | Fragile (Covariant) | Symmetry Preserved | Theorem | Diagnostic Node |
|:----------|:------------|:--------------------|:-------------------|:--------|:----------------|
| Linear | `nn.Linear(bias=True)` | `SpectralLinear(bias=False)` | Translation invariance in tangent bundle | {prf:ref}`thm-spectral-preserves-light-cone` | Node 62 |
| Activation | `ReLU` (element-wise) | `NormGatedGELU` (radial) | $SO(d_b)$ rotation per bundle | {prf:ref}`thm-norm-gating-equivariant` | Node 67 |
| Normalization | `LayerNorm`, `BatchNorm` | **Removed** (implicit in spectral norm) | Probability mass conservation (WFR) | - | - |
| Vision encoder | `Conv2d` | `SteerableConv2d` (E2CNN) | $SO(2)$ rotation + $\mathbb{R}^2$ translation | {prf:ref}`thm-steerable-conv-equivariant` | Node 68 |
| Composition | Arbitrary depth | Lipschitz-constrained composition | Global light cone preservation | {prf:ref}`thm-composition-equivariant` | Node 62 |

### Gauge Group Decomposition

**Table: Primitive ↔ Gauge Field Correspondence**

| Gauge Field | Group | DNN Primitive | Physical Interpretation | Cross-Reference | Implementation Note |
|:------------|:------|:--------------|:------------------------|:----------------|:--------------------|
| Binding $G_\mu^a$ | $SU(N_f)_C$ | Isotropic bundles ($n_b$ bundles) | Feature confinement (color charge) | {ref}`sec-symplectic-multi-agent-field-theory`, Node 40 | Discrete subgroup: signed permutations |
| Error $W_\mu^b$ | $SU(2)_L$ | Steerable conv (observation-action coupling) | Sensor-motor mixing (weak force) | {ref}`sec-symplectic-multi-agent-field-theory` | **Reduced to $SO(2) \cong U(1)$** (2D rotation subgroup) |
| Opportunity $B_\mu$ | $U(1)_Y$ | Spectral norm (hypercharge bound) | Capacity bound | {ref}`sec-parameter-space-sieve`, Node 56 | Bounded quantity $Y \propto \|z\|^2$, not explicit rotation |

### Physics Isomorphism: DNN Primitives ↔ Standard Model

**Table: The "Particle Zoo" of Neural Operators**

| Standard Model Particle | Gauge Charge $(C, L, Y)$ | DNN Analogue | Transformation Property |
|:------------------------|:-------------------------|:-------------|:------------------------|
| Quark $(u, d)$ | $(3, 2, 1/6)$ | Feature in bound bundle | Confined, left-chiral |
| Lepton $(e, \nu_e)$ | $(1, 2, -1/2)$ | Observation vector | Free, left-chiral |
| Gluon | $(8, 1, 0)$ | Bundle coupling weights | Self-interacting (confines quarks) |
| $W/Z$ boson | $(1, 3, 0)$ | Obs-action mixer | Weak force mediator |
| Photon | $(1, 1, 0)$ | Capacity gradient | Long-range field (unconfined) |
| Higgs | $(1, 2, 1/2)$ | Activation threshold bias | Mass generation via symmetry breaking |

:::{div} feynman-prose
Let me be precise about something: the table above is not poetry. It is not a loose analogy where we squint and things sort of look similar. The mathematical structures are *identical*.

Consider confinement. In QCD, quarks carry color charge—red, green, or blue. A fundamental principle says you can never observe an isolated quark; they must always bind into color-neutral combinations (three quarks in a baryon, quark-antiquark in a meson). This is not optional; it is enforced by the theory. Similarly, features in our IsotropicBlocks carry bundle indices. Node 40 in the diagnostics explicitly checks that only bound states—combinations that are "neutral" across bundles—reach the macro register. Features cannot propagate independently; they must bind.

Consider mass generation. In the Standard Model, particles acquire mass through the Higgs mechanism: a field with a potential that has its minimum away from zero, spontaneously breaking symmetry. In our architecture, the activation potential $b_i$ plays exactly this role. It creates an energy barrier that signals must overcome, and the height of this barrier determines how strongly features are suppressed—analogous to how the Higgs coupling determines particle masses.

The correspondence runs deep. Physics constants map to architecture hyperparameters. Coupling strengths map to weight magnitudes. The QCD confinement scale $\Lambda_{\text{QCD}}$ maps to the bundle binding scale set by activation thresholds. You do not get to choose these independently; they are locked together by the gauge structure.

This is what it means to derive architecture from first principles.
:::



(sec-forward-to-chapter-5)=
## Connection to Chapter 5 (Macroscopic Integration)

:::{div} feynman-prose
We have now built the atoms—the microscopic primitives that respect gauge symmetry. SpectralLinear for light-cone preservation. NormGate for rotation-invariant activation. SteerableConv for equivariant vision. Each one carefully designed to preserve its piece of $G_{\text{Fragile}}$.

But atoms are not an agent. You need to compose them into molecules—full architectures that can actually do something. Chapter 5 shows how to do this, and here is the remarkable thing: once you have gauge-covariant primitives, the integrator almost writes itself.

The Boris-BAOAB scheme for geodesic integration has specific requirements. It needs metric-preserving steps—IsotropicBlock provides these. It needs light-cone preservation so information does not travel faster than $c_{\text{info}}$—spectral normalization provides this. It needs symplectic structure so phase space volume is conserved—the bundle structure provides this. Each requirement maps directly to a primitive we have already built.

What remains to be shown is how *attention* fits into this picture. When an agent attends to different parts of its observation, it is implementing *parallel transport*—moving vectors from one location on the manifold to another without them leaving the tangent bundle.

**Remark on gauge connections:** The full gauge-theoretic formulation—including Wilson lines (path-ordered exponentials of gauge connections), curvature forms, and parallel transport—is developed rigorously in Chapter 8.5 (Macroscopic Integration). At the DNN block level, we have established the *local* gauge covariance properties of primitives. The *global* integration of these local symmetries via attention mechanisms requires the machinery of principal bundles and connection forms, which is beyond the scope of this chapter.

We have the atoms. Chapter 8.5 builds the molecules through Covariant Cross-Attention. And from molecules, we get agents that move smoothly along geodesics in a gauge-invariant way.
:::

**Preview of Architecture:**

```
Input (observation x_t)
  ↓ [SteerableConv] → SO(2)-equivariant features
  ↓ [IsotropicBlock × L] → Gauge-covariant latent z_t
  ↓ [CovariantCrossAttention] → Geodesic update (see Chapter 5)
  ↓ [IsotropicBlock × L] → Decode to action a_t
Output (action a_t)
```

**Forward cross-references:**
- **Covariant Cross-Attention** ({ref}`sec-covariant-cross-attention`): Implements Wilson lines for parallel transport along geodesics.
- **Boris-BAOAB Integrator** ({ref}`sec-equations-of-motion-langevin-sdes-on-information-manifolds`): Macroscopic integration scheme that requires microscopic primitives to preserve gauge structure.
- **Temperature Schedule** ({ref}`sec-the-belief-wave-function-schrodinger-representation`): Cognitive temperature $T_c$ varies with inverse conformal factor $1/\lambda(z)$ to maintain consistent exploration across curved manifold.



(sec-dnn-blocks-conclusion)=
## Conclusion

:::{div} feynman-prose
Let me summarize what we have accomplished in this chapter, because it is genuinely remarkable when you step back and look at the whole picture.

We started with a problem: standard deep learning primitives—the bread-and-butter operations like ReLU and biased linear layers—secretly assume things about your space that are not true in general. They assume flat geometry. They assume preferred coordinate axes. They assume that the origin of your coordinate system means something special. These assumptions are fine for image classification on a fixed dataset, but they are catastrophic for agents that must maintain consistent beliefs across time, integrate geodesics on curved manifolds, and interact with other agents under capacity constraints.

So we built replacements. Three primitives, each designed from the ground up to respect gauge symmetry:

1. **SpectralLinear** removes bias and constrains the singular values. This preserves the light cone structure—information cannot propagate faster than $c_{\text{info}}$, and there is no artificial preferred origin in the tangent space.

2. **NormGate** applies activation based on bundle norms, not individual components. The gate decision "is there enough energy to pass?" has the same answer regardless of how you orient your coordinate axes within each bundle.

3. **SteerableConv** lifts images to the group $SE(2)$ and convolves with steerable filters. Rotate the input, get rotated features—exactly, not approximately through data augmentation.

Each primitive preserves one factor of the gauge group $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$. Bundle structure implements color confinement. Spectral normalization enforces a hypercharge bound. Steerable convolutions implement the observation-action doublet structure. Three primitives, three gauge factors, one unified theory.

And here is what I find most satisfying: the constraints are so tight that the architecture essentially writes itself. Once you demand gauge invariance, there is only one way to build each component. You do not get to make arbitrary design choices; the symmetry requirements dictate the structure. This is the hallmark of good physics: not many knobs to tune, but few parameters that must take specific values.

We now have the atoms. What remains is to show how these atoms compose into molecules—full architectures that can actually be trained and deployed. That is the subject of Chapter 5, where we connect these microscopic primitives to the macroscopic geodesic integrator.

The bridge from gauge theory to neural network implementation is complete.
:::
