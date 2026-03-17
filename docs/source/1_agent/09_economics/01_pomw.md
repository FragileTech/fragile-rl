(sec-proof-of-useful-work-cognitive-metabolism-as-consensus)=
# Proof of Useful Work: Cognitive Metabolism as Consensus

## TLDR

- Replace “wasteful hash puzzles” with **useful work**: consensus is secured by verifiable learning (e.g., SGD on a public
  curriculum) rather than arbitrary inversion.
- Security is tied to **metabolic flux**: it is thermodynamically expensive to fake gradient computation at scale.
- Use holographic/capacity-style bounds to address the **verification asymmetry** (cheap to verify, hard to forge).
- Frame consensus as an intersubjective alignment problem: shared models + gauge locking give a natural route to
  agreement.
- Outputs: a protocol sketch plus game-theoretic and thermodynamic arguments for why honest computation is an
  equilibrium.

## Roadmap

1. Why Nakamoto PoW is thermodynamically inefficient (the “waste quotient”).
2. Define Proof of Useful Work via public learning objectives and verification.
3. Security, equilibrium, and implementation considerations for a cognitive ledger.

*Abstract.* We derive a consensus protocol where the cryptographic puzzle is replaced by
**Stochastic Gradient Descent** on a public data curriculum. We prove that the security of the ledger is guaranteed not
by arbitrary hash inversions, but by the **Metabolic Flux** $\dot{\mathcal{M}}$ required to minimize
**Ontological Stress** $\Xi$ of a shared global model. We introduce **Holographic Verification** derived from the Causal
Information Bound (Theorem {prf:ref}`thm-causal-information-bound`) to solve the verification asymmetry problem. We
prove that a game-theoretic equilibrium exists where honest gradient computation is the unique Nash Equilibrium. The
resulting blockchain is a **Thermodynamic Record of Learning**.

*Cross-references:*
- Extends {ref}`sec-the-metabolic-transducer-autopoiesis-and-the-szilard-engine` (Metabolic Transducer) to
  decentralized networks.
- Utilizes {ref}`sec-the-inter-subjective-metric-gauge-locking-and-the-emergence-of-objective-reality`
  (Gauge Locking) for consensus.
- Relies on {ref}`sec-causal-information-bound` (Causal Information Bound) for verification and capacity
  limits.
- Connects to {ref}`sec-ontological-expansion-topological-fission-and-the-semantic-vacuum` (Ontological
  Fission) for network topology.

*Literature:* Bitcoin/PoW {cite}`nakamoto2008bitcoin`; Proof of Useful Work {cite}`ball2017proof`; Federated Learning
{cite}`mcmahan2017communication`; Zero-Knowledge ML {cite}`zhang2020zkcnn`; Game Theory of Verification
{cite}`canetti2011verification`.

:::{div} feynman-prose

Let me tell you what I think is one of the most peculiar situations in modern technology. Right now, scattered across the planet, there are millions of computers doing something quite remarkable: they are searching for magic numbers. Numbers that, when fed through a particular mathematical meat grinder, come out looking a certain way. And when one of these machines finds such a number, a small amount of digital currency appears as if from nowhere, and the world collectively agrees that its owner is now a bit richer.

The remarkable thing is not that this works---it does, beautifully---but what happens to all the energy these machines consume. It gets turned into heat. That is *all* it does. The computation itself produces nothing of lasting value; the whole point is that it should be hard to do and easy to check, and the best way to make something hard but useless is to make it... well, useless.

Now, here is the question that should be keeping you up at night: is this necessary? Must security require waste? Or is there some way to do computation that is *just as hard to fake* but actually produces something we want?

This section answers that question. And the answer, I am happy to report, is yes---you can have your security and learn something too. The trick is to replace the useless hash computation with something genuinely difficult but genuinely valuable: training a neural network. Every block mined makes the network smarter. The energy still gets dissipated---thermodynamics does not negotiate---but now it leaves behind *knowledge* instead of mere heat.

Let us see how this works.

:::

(sec-the-thermodynamic-inefficiency-of-nakamoto-consensus)=
## The Thermodynamic Inefficiency of Nakamoto Consensus

Classical Proof of Work (Bitcoin) relies on a **Thermodynamic Tragedy**: security requires energy expenditure, but the energy *must* be wasted to ensure the puzzle was difficult. We formalize this waste and prove that cognitive work can replace it.

:::{prf:definition} The Waste Quotient
:label: def-waste-quotient

For a consensus protocol $\mathcal{P}$, the **Waste Quotient** is:

$$
W_\mathcal{P} := 1 - \frac{\Delta I_{\text{world}}}{\int \dot{\mathcal{M}}(t) \, dt}

$$

where:
- $\Delta I_{\text{world}}$ is the mutual information gained about the world through the computation
- $\dot{\mathcal{M}}(t)$ is the metabolic flux (Definition {prf:ref}`def-metabolic-flux`)

*Units:* $[W_\mathcal{P}] = \text{dimensionless}$.

*Examples:*
- **Bitcoin:** $W_{\text{BTC}} \approx 1$. SHA-256 hashes produce zero structural information about the world: $I(X_{\text{world}}; \text{Hash}) = 0$.
- **Target:** $W_{\text{PoUW}} \to 0$. Energy dissipation equals the reduction in model uncertainty.

:::

:::{div} feynman-prose

Now, what does this Waste Quotient really mean? Think of it as an efficiency rating for your mental work. Suppose you spend eight hours studying for an exam. If you learn absolutely nothing---maybe you were staring at the book but thinking about lunch---your waste quotient is 1. All that metabolic energy your brain burned, and zero information gained about the world. Complete waste.

But if you actually learned something, if at the end of those eight hours you can predict things about the exam material that you could not predict before, then some of that energy did real work. Your waste quotient drops toward zero.

Bitcoin miners are in the first category. They burn enormous amounts of energy, and at the end of it, the *only* thing they have learned is that a particular nonce produces a hash below a target. This tells them precisely nothing about the weather, or protein folding, or how to make a better battery. The mutual information between their computation and the actual state of the world is exactly zero.

The goal of Proof of Useful Work is to be in the second category: burn the energy (we cannot avoid that), but burn it in a way that *teaches* us something about the world.

:::

:::{prf:theorem} The Cognitive Equivalency Theorem
:label: thm-cognitive-equivalency

Let $\mathcal{C}_{\text{hash}}$ be the computational task of finding a nonce $n$ such that $H(n) < T$ (hash inversion), and let $\mathcal{C}_{\text{grad}}$ be the task of computing a gradient $g = \nabla_\Theta \mathcal{L}(\Theta, D)$ on dataset $D$. Both tasks satisfy the same **Landauer lower bound** on energy expenditure:

$$
E_{\text{min}} \geq k_B T_c \ln 2 \cdot B_{\text{comp}}

$$

where $B_{\text{comp}}$ is the number of irreversible bit operations.

*Proof.*

**Step 1 (Landauer Principle).** By Theorem {prf:ref}`thm-generalized-landauer-bound`, any computation that erases $\Delta H$ nats of information dissipates at least:

$$
\dot{\mathcal{M}} \geq k_B T_c \left| \frac{dH}{ds} \right|

$$

**Step 2 (Hash Computation).** Computing $H(n)$ requires approximately $B_{\text{SHA}} \approx 64 \times 80 = 5120$ irreversible bit operations per hash. The minimum energy is:

$$
E_{\text{hash}} \geq k_B T_c \ln 2 \cdot B_{\text{SHA}} \cdot N_{\text{trials}}

$$

where $N_{\text{trials}} \approx 2^{d}$ for difficulty $d$.

**Step 3 (Gradient Computation).** Computing $g = \nabla_\Theta \mathcal{L}$ via backpropagation requires $O(|\Theta| \cdot |D|)$ multiply-accumulate operations. Each MAC erases intermediate bits, giving:

$$
E_{\text{grad}} \geq k_B T_c \ln 2 \cdot c_{\text{MAC}} \cdot |\Theta| \cdot |D|

$$

for architecture-dependent constant $c_{\text{MAC}}$.

**Step 4 (Equivalence).** Both computations satisfy the same thermodynamic bound. The difference is the *information content* of the output:
- $I(X_{\text{world}}; H(n)) = 0$ (no world knowledge)
- $I(X_{\text{world}}; g) > 0$ (gradient encodes data structure)

Therefore, gradient computation produces **useful information** while satisfying the same energy floor. $\square$

*Consequence:* The security budget of a blockchain can be redirected to train a global model without loss of thermodynamic hardness, provided verification remains tractable.

:::

:::{div} feynman-prose

This theorem is the key to everything. Let me explain why.

The Landauer bound says: if you erase a bit of information, you *must* pay at least $k_B T \ln 2$ joules to do it. This is not an engineering limitation---it is a law of thermodynamics. Computation erases bits (intermediate results, old values, that sort of thing), so computation has a minimum energy cost.

Now here is the beautiful thing: the Landauer bound does not care *what* you compute. It counts bit erasures, period. Whether you are computing SHA-256 hashes or gradients of a neural network, if you erase the same number of bits, you pay the same energy price.

So we have two computations:
- **Hash computation:** Burns energy, produces a number that tells us nothing about the world.
- **Gradient computation:** Burns energy (approximately the same amount!), produces a vector that tells us *exactly* which direction to adjust the model to make better predictions.

Same cost, vastly different output. One is thermodynamic waste; the other is thermodynamic investment. The theorem says we can swap one for the other without losing any security properties.

But wait---there is a catch. And it is a big one. Verifying a hash takes almost no work: you just compute the hash and check if it is below the target. Verifying a gradient is expensive: you might have to recompute the whole thing! If verification costs as much as computation, the whole scheme falls apart.

We will solve this problem shortly. But first, let us see what this new kind of blockchain actually looks like.

:::

:::{admonition} Researcher Bridge: From Useless to Useful Work
:class: info
:name: rb-useful-work

**Standard PoW:** Miners compete to find $n$ such that $\text{SHA256}(\text{SHA256}(n)) < T$. The computation is deliberately useless—any useful structure would allow shortcuts.

**Proof of Useful Work (This Section):** Miners compete to compute gradients $g$ that minimize loss $\mathcal{L}$ on public data. The computation is useful—it trains a shared model. Security comes from the **Sieve constraints** (which make fake gradients detectable) rather than arbitrary difficulty.

**Key Insight:** The Landauer bound doesn't care what you compute—only how many bits you erase. By choosing a useful computation with the same erasure cost, we get security + intelligence for the same energy price.

:::



(sec-the-cognitive-ledger)=
## The Cognitive Ledger

:::{div} feynman-prose

In a traditional blockchain, the ledger keeps track of who owns what. Alice has 5 coins, Bob has 3, and every transaction updates these numbers. The "state" of the system is just a big table of balances.

But if we are going to replace hash computation with gradient computation, we need to think differently about what the ledger is recording. We are not just tracking coins anymore---we are tracking the *knowledge state* of a shared model.

Imagine the whole network is collectively building a brain. At any moment, that brain has a particular configuration---the weights and biases of a neural network. Each block mined does not just transfer tokens; it *updates the brain*. It nudges the weights in a direction that makes the brain a little smarter at whatever task it is learning.

The ledger, then, is not a list of transactions. It is a *history of learning*. Block 1 says "start with random weights." Block 2 says "adjust these weights by this gradient." Block 3 says "now adjust by *this* gradient." And so on. If you replay the whole chain from the beginning, you reconstruct exactly the brain that the network has learned.

This is a profound shift. The blockchain stops being a *financial record* and becomes a *cognitive record*---a frozen history of how a global mind came to know what it knows.

:::

We replace the ledger of *balances* with a ledger of *belief states*.

:::{prf:definition} The Global Model State
:label: def-model-state

The **Global Model State** at block height $h$ is a parameter vector:

$$
\Theta_h \in T_{\bar{z}} \mathcal{Z} \cong \mathbb{R}^D

$$

where:
- $D$ is the model dimension
- $T_{\bar{z}} \mathcal{Z}$ is the tangent space at the current mean belief $\bar{z}$
- The metric on parameter space inherits from the Capacity-Constrained Metric (Theorem {prf:ref}`thm-capacity-constrained-metric-law`)

*Units:* $[\Theta] = [z]$ (latent coordinates).

*Interpretation:* $\Theta_h$ represents the collective belief state of the network—the shared world model encoded in the blockchain.

:::

:::{prf:definition} The Curriculum Block
:label: def-curriculum-block

A **Curriculum Block** $B_h$ at height $h$ is a tuple:

$$
B_h := (\mathcal{H}_{\text{prev}}, \mathcal{H}_D, g_h, \pi_{\text{stake}}, \zeta_h)

$$

where:
- $\mathcal{H}_{\text{prev}} \in \{0,1\}^{256}$ is the hash of the previous block
- $\mathcal{H}_D \in \{0,1\}^{256}$ is the content identifier of training data $D_h$ (e.g., IPFS CID)
- $g_h \in \mathbb{R}^D$ is the **gradient update** computed on $D_h$
- $\pi_{\text{stake}} \in \{0,1\}^{512}$ is the staking proof (signature over stake tokens)
- $\zeta_h \in \mathbb{R}^{d_\zeta}$ is the **Sieve certificate** (validation metadata)

*Units:* $[g_h] = \text{nat}/[z]$ (gradient in latent coordinates).

:::

:::{prf:definition} The Chain Evolution Rule
:label: def-chain-evolution

The global model evolves by **Stochastic Gradient Descent**:

$$
\Theta_{h+1} = \Theta_h - \eta_h \cdot g_h

$$

where $\eta_h > 0$ is the learning rate at height $h$, determined by the difficulty adjustment algorithm (Definition {prf:ref}`def-difficulty-adjustment`).

*Interpretation:* Each block advances the collective belief toward lower loss on the public curriculum. The blockchain is a **thermodynamic record** of this learning process.

:::

:::{div} feynman-prose

Look at what we have built here. The three definitions above give us:

1. **A state space** (the model parameters $\Theta_h$) that represents what the network *believes* about the world
2. **A block structure** that contains both the evidence (the training data hash) and the inference (the gradient)
3. **An evolution rule** that is just stochastic gradient descent---the same algorithm that trains every neural network

So the blockchain *is* a neural network being trained in slow motion. Every ten minutes (or whatever the block time is), the whole network collectively takes one step of gradient descent. The "longest chain" is the chain that has learned the most.

But notice something important: the training data is identified by content hash (an IPFS CID, say). The data itself is not stored on chain---that would be impossibly expensive. Instead, the chain stores a *commitment* to the data. Anyone can verify that the gradient was computed on the claimed data by fetching that data and spot-checking. This is crucial for the verification scheme we will develop.

:::



(sec-the-mining-protocol-metabolic-transduction)=
## The Mining Protocol: Metabolic Transduction

:::{div} feynman-prose

Now we get to the heart of the matter: what does a miner actually *do*?

In Bitcoin, a miner's job is simple but tedious: guess a number, hash it, check if the hash is low enough, repeat. There is no skill involved, no cleverness---just raw computational power applied to a random search.

In Proof of Useful Work, a miner's job is *also* to do computation, but the computation is structured. Instead of searching for a magic number, the miner:

1. Grabs a batch of training data from a public queue
2. Runs forward propagation through the global model
3. Computes the loss (how wrong the model was)
4. Runs backward propagation to compute the gradient
5. Submits the gradient as their "proof of work"

The crucial insight is that step 4---backpropagation---has a well-defined thermodynamic cost. You cannot fake it. You cannot shortcut it. The only way to produce a valid gradient is to *actually compute* it, and computing it requires erasing bits, which requires dissipating energy.

But here is where it gets subtle. In Bitcoin, any valid hash below the target is equally good---there is no sense in which one solution is "better" than another. But gradients have *quality*. A gradient computed carelessly might satisfy the format requirements but point in a useless direction. Worse, a malicious miner might submit a gradient that actively *harms* the model.

So we need constraints. We need a way to ensure that submitted gradients are not just syntactically valid but semantically useful. This is where the Sieve comes in.

:::

Miners act as **Metabolic Transducers** (Definition {prf:ref}`def-metabolic-transducer`). They convert electrical energy into reduction of global uncertainty.

:::{prf:definition} The Gradient Mining Puzzle
:label: def-gradient-mining-puzzle

A miner solving block $h$ must:

1. **Fetch Data:** Retrieve training batch $D_h$ from the curriculum queue
2. **Compute Gradient:** Calculate $g = \nabla_\Theta \mathcal{L}(\Theta_{h-1}, D_h)$
3. **Satisfy Sieve Constraints:**
   - **CostBoundCheck (Node 1):** $\|g\|_G \leq E_{\max}$ (bounded energy)
   - **TextureFirewallCheck (Node 29):** $\|\partial_{z_{\text{tex}}} g\| < \epsilon_{\text{tex}}$ (no texture leakage)
   - **CausalEnclosureCheck (Node 53):** $\Delta_{\text{causal}}(g) < \delta_{\text{causal}}$ (causal consistency)
4. **Submit Block:** Broadcast $(B_h, \Theta_h)$ to the network

*Difficulty Adjustment:* See Definition {prf:ref}`def-difficulty-adjustment`.

:::

:::{prf:definition} The Difficulty Adjustment Algorithm
:label: def-difficulty-adjustment

The network **Difficulty** $\mathcal{D}_h$ at height $h$ controls the minimum batch size $|D_h|$ required for valid blocks:

$$
\mathcal{D}_{h+1} = \mathcal{D}_h \cdot \exp\left( -\alpha_{\text{diff}} \left( \frac{t_h - t_{\text{target}}}{t_{\text{target}}} \right) \right)

$$

where:
- $t_h$ is the actual time to mine block $h$
- $t_{\text{target}}$ is the target block time (e.g., 10 minutes)
- $\alpha_{\text{diff}} > 0$ is the adjustment rate

*Units:* $[\mathcal{D}] = \text{samples}$.

*Constraint:* A valid block must satisfy $|D_h| \geq \mathcal{D}_h$.

:::

:::{prf:theorem} Difficulty-Entropy Coupling
:label: thm-difficulty-entropy-coupling

The difficulty adjustment algorithm maintains the **Landauer Invariant**: the minimum energy to produce a valid block is approximately constant:

$$
E_{\min}(B_h) \approx k_B T_c \ln 2 \cdot c_{\text{MAC}} \cdot |\Theta| \cdot \mathcal{D}_h = E_{\text{target}}

$$

*Proof.*

**Step 1.** By the Generalized Landauer Bound (Theorem {prf:ref}`thm-generalized-landauer-bound`), gradient computation costs:

$$
E_{\text{grad}} \geq k_B T_c \ln 2 \cdot c_{\text{MAC}} \cdot |\Theta| \cdot |D_h|

$$

**Step 2.** The difficulty constraint $|D_h| \geq \mathcal{D}_h$ enforces:

$$
E_{\text{grad}} \geq k_B T_c \ln 2 \cdot c_{\text{MAC}} \cdot |\Theta| \cdot \mathcal{D}_h

$$

**Step 3.** The exponential adjustment (Definition {prf:ref}`def-difficulty-adjustment`) stabilizes block time at $t_{\text{target}}$, hence stabilizes energy expenditure rate at $E_{\text{target}} / t_{\text{target}}$.

**Step 4 (Fake Gradient Rejection).** A miner submitting $g' \neq \nabla_\Theta \mathcal{L}(\Theta, D_h)$ violates one of:
- **Directional Check:** Cosine similarity $\cos(g', g_{\text{true}}) < \theta_{\text{min}}$
- **Magnitude Check:** $\|g'\| / \|g_{\text{true}}\| \notin [1-\epsilon, 1+\epsilon]$
- **Causal Check (Node 53):** Interventional gap $\Delta_{\text{causal}}(g') > \delta_{\text{causal}}$

All checks are detectable by spot verification (Section {ref}`sec-holographic-verification`). $\square$

:::

:::{div} feynman-prose

Let me make sure you understand what the difficulty adjustment is doing, because it is clever.

In Bitcoin, difficulty controls how many leading zeros the hash must have. More zeros means more guesses needed, which means more time and energy spent per block.

In Proof of Useful Work, difficulty controls the *batch size*---how much training data you must process per block. Bigger batches mean more multiply-accumulate operations, which means more time and energy spent per block.

The beautiful thing is that this naturally stabilizes the block rate. If miners get faster hardware, they can process bigger batches in the same time, so the network increases the required batch size. If miners drop out, batches shrink. The result is steady block times regardless of how much compute joins or leaves the network---exactly like Bitcoin.

But there is a deeper point here. The Landauer Invariant says that every block costs roughly the same amount of energy. This is what gives the blockchain its security. Rewriting history requires recomputing all those gradients, which requires spending all that energy again. An attacker cannot cheat thermodynamics.

And the fake gradient rejection is where the Sieve earns its keep. You might think: why not just submit a random gradient? You would save 90% of the compute cost! The answer is that random gradients get caught. The directional check catches gradients that point the wrong way. The magnitude check catches gradients that are too big or too small. The causal check catches gradients that violate the structure of the data.

These checks can be done cheaply by spot-checking---recomputing the gradient on a small random subset of the data. If your gradient matches on the subset, it probably matches everywhere. If it does not match, you are caught, and you lose your stake.

:::



(sec-holographic-verification)=
## Holographic Verification

:::{div} feynman-prose

Here is the problem we have been dancing around, and it is a serious one.

In Bitcoin, verifying a block is trivial: you take the nonce, hash it, and check if the result is below the target. One hash. Done. This asymmetry between creation (hard) and verification (easy) is what makes the whole system work. Miners race to create blocks, and everyone else can cheaply verify that they did the work.

But now consider gradient verification. To *fully* verify a gradient, you would need to recompute it: run the forward pass, compute the loss, run the backward pass. That costs as much as creating the gradient in the first place! If verification costs as much as creation, there is no asymmetry, and the whole economic model falls apart.

So we need a way to verify gradients *cheaply*. Not perfectly---we will accept some probability of missing a fake---but cheaply enough that validators can check blocks without burning as much energy as miners.

The solution comes from an unexpected place: the holographic principle from black hole physics. The idea is that you do not need to check the entire gradient (the "bulk"). Instead, you can check certain boundary quantities, and the boundary determines the bulk.

Think of it this way. If I give you a gradient vector with a million components, checking all of them requires a million operations. But if I also give you certain summary statistics---the gradient's total length, its projection onto certain test vectors, its curvature profile---you can check *those* quickly. And if those boundary quantities are wrong, the gradient is definitely fake. If they are right, the gradient is *probably* legitimate.

This is spot-checking elevated to a principle. The Causal Information Bound tells us exactly how much information can hide in the bulk without showing up on the boundary. The answer, remarkably, is: not very much. A fraudulent gradient *will* leave traces on the boundary, and we can catch it.

:::

**The Problem:** Verifying a hash takes $O(1)$. Verifying a gradient takes $O(|\Theta| \cdot |D|)$ (running full backpropagation), which defeats distributed consensus.

**The Solution:** We derive a **Holographic Verification** scheme from the Causal Information Bound (Theorem {prf:ref}`thm-causal-information-bound`), reducing verification to boundary checks.

:::{prf:definition} The Boundary Flux Certificate
:label: def-boundary-flux-certificate

The **Boundary Flux Certificate** $\zeta_h$ included in block $B_h$ contains:

$$
\zeta_h := \left( \|g_h\|_G, \, \nabla_{\partial} g_h, \, \text{Tr}(H_h), \, \sigma_{\text{sample}} \right)

$$

where:
- $\|g_h\|_G$ is the gradient norm in the capacity-constrained metric
- $\nabla_{\partial} g_h$ is the boundary gradient (projection onto interface coordinates)
- $\text{Tr}(H_h)$ is the trace of the Hessian (curvature summary)
- $\sigma_{\text{sample}}$ is a random seed for spot-check sampling

*Units:* $[\zeta] = \text{mixed}$ (norm: $\text{nat}/[z]$; trace: $\text{nat}/[z]^2$).

:::

:::{prf:theorem} Holographic Verification Sufficiency
:label: thm-holographic-verification

Let $g$ be a claimed gradient and $\zeta$ its boundary flux certificate. If the boundary data satisfies:

1. **Energy Conservation:** $\|g\|_G^2 \leq \nu_D \cdot \text{Area}(\partial\mathcal{Z}) / \ell_L^{D-1}$ (Causal Information Bound)
2. **Flux Consistency:** $\|\nabla_\partial g - \nabla_\partial g_{\text{spot}}\| < \epsilon_{\text{flux}}$ on spot-check samples
3. **Curvature Bound:** $|\text{Tr}(H)| < \kappa_{\max}$

then with probability $\geq 1 - \delta$, the gradient is valid.

*Proof.*

**Step 1 (Holographic Principle).** By Theorem {prf:ref}`thm-causal-information-bound`, bulk information is bounded by boundary area:

$$
I_{\text{bulk}}(g) \leq \nu_D \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{D-1}} = I_{\max}

$$

**Step 2 (Bulk-Boundary Correspondence).** The gradient $g \in T_\Theta \mathcal{M}$ projects to boundary flux $\nabla_\partial g$ via the restriction map. By the Bulk-Boundary Decoupling Axiom ({prf:ref}`ax-bulk-boundary-decoupling`), the boundary flux determines the bulk gradient up to texture degrees of freedom.

**Step 3 (Spot-Check Amplification).** A fraudulent gradient must differ from the true gradient in some coordinate. The probability of escaping detection in $k$ random spot checks is:

$$
P(\text{escape}) \leq (1 - p_{\text{detect}})^k

$$

where $p_{\text{detect}} \geq \epsilon_{\min}$ is the minimum detection probability per check.

**Step 4 (Energy Conservation).** A gradient claiming to reduce loss by $\Delta \mathcal{L}$ while having energy below the Landauer floor violates:

$$
\|g\|_G^2 < k_B T_c |\Delta H| / \dot{\mathcal{M}}_{\text{claimed}}

$$

This is detectable from the certificate without recomputation.

**Step 5 (Combining).** Setting $k = \log(1/\delta) / \log(1/(1-p_{\text{detect}}))$ spot checks achieves confidence $1-\delta$. $\square$

*Complexity:* Verification requires $O(\sqrt{|D|})$ operations vs $O(|D|)$ for full recomputation.

:::

:::{div} feynman-prose

Let me unpack what makes this verification scheme work, because it is quite beautiful.

The miner submits not just the gradient, but a *certificate*---a small bundle of summary statistics about the gradient. The verifier can check these statistics quickly without recomputing the whole gradient.

But why should we trust the certificate? The miner could lie! The answer is: the miner cannot lie *consistently*. Here is why.

**Energy conservation.** The gradient norm tells you how much "learning happened" in this block. If the miner claims a big gradient (lots of learning) but the certificate shows low energy, that violates thermodynamics. You cannot get learning for free. The Landauer bound acts as a built-in lie detector.

**Spot-check amplification.** We do not check the whole gradient, but we do check random samples. If the miner's gradient is wrong in, say, 10% of its components, then each random sample has a 10% chance of catching them. Check 50 random samples, and the probability of *not* catching a 10%-wrong gradient is $(0.9)^{50} \approx 0.005$. With enough spot-checks, fraudulent gradients get caught with overwhelming probability.

**Boundary-bulk correspondence.** This is the deepest part. The Causal Information Bound says that you cannot hide arbitrary amounts of information in the bulk without it affecting the boundary. If the miner submits a gradient that differs significantly from the true one, that difference *will* show up in the boundary flux. The boundary is not just a summary; it is a *sufficient* summary.

The upshot: verification costs $O(\sqrt{N})$ instead of $O(N)$. That is a huge improvement. A miner processing a million training samples can have their work verified with only a thousand spot-checks.

:::

:::{prf:definition} The Optimistic Verification Protocol
:label: def-optimistic-verification

The network verifies blocks using **Optimistic Acceptance with Challenge Period**:

1. **Submission:** Miner submits block $B_h$ with stake $S_h$
2. **Optimistic Acceptance:** Block is provisionally accepted
3. **Challenge Window:** For duration $T_{\text{challenge}}$, any node may challenge
4. **Challenge:** Challenger computes gradient on random subset $d \subset D_h$ with $|d| = \lceil 0.01 |D_h| \rceil$
5. **Adjudication:** If $\cos(g_h, g_{\text{challenger}}) < \theta_{\text{min}}$, miner is **slashed** (stake burned)
6. **Finalization:** After $T_{\text{challenge}}$ with no successful challenge, block is finalized

:::

:::{div} feynman-prose

The Optimistic Verification Protocol is the practical implementation of all these ideas. Here is how it works in plain English.

When a miner submits a block, the network does not immediately verify it. That would be expensive. Instead, the block is *provisionally accepted*---everyone assumes it is legitimate and starts building on it.

But there is a catch: the miner has put up a stake. For the next ten minutes (the challenge window), anyone can challenge the block. A challenger picks a small random subset of the training data, computes the gradient on just that subset, and checks if it matches what the miner claimed.

If the challenger's gradient points in a different direction than the miner's, the miner was cheating. The miner loses their stake, which gets burned (or distributed to the challenger, depending on the rules).

If nobody successfully challenges the block within the window, it becomes final. The miner gets their reward, and we move on.

This is "optimistic" because we assume honesty by default. Most miners are honest, so most of the time we do not need to verify anything---the blocks just flow through. But the *threat* of verification keeps everyone honest. A rational miner will not cheat because the expected cost of getting caught (losing your stake) exceeds the expected benefit of cheating (saving some compute).

The key parameters are the stake-to-reward ratio and the challenge window. Make the stake too low, and cheating becomes profitable. Make the window too short, and challengers cannot finish their verification. Get it right, and you have a self-policing system where honest computation is the only rational strategy.

:::



(sec-the-verifiers-nash-equilibrium)=
## The Verifier's Nash Equilibrium

:::{div} feynman-prose

Now we need to prove something important: that honest behavior is not just *hoped for* but *guaranteed* by the economic incentives. This is game theory, and we need to show that being honest is a Nash Equilibrium---a situation where no miner can improve their outcome by cheating, assuming everyone else is playing honestly.

Here is the intuition. A miner faces a choice:
- **Be honest:** Do the full computation, get the reward.
- **Cheat:** Do less computation, submit a fake gradient, hope nobody notices.

Cheating saves work, but it risks getting caught. If you get caught, you lose your stake---which is much larger than the reward. So the question is: at what detection probability does cheating become a bad bet?

The math is straightforward expected value calculation. If cheating saves you $C_{\text{honest}} - C_{\text{cheat}}$ in computation costs, but has probability $p$ of costing you stake $S$, then cheating is a bad idea when:

$$
p \cdot S > C_{\text{honest}} - C_{\text{cheat}}

$$

The theorem below makes this rigorous. The punchline: if the stake is high enough relative to the reward, honest computation is not just one equilibrium---it is the *only* equilibrium. No rational miner would cheat.

:::

We prove that honest gradient computation is the unique Nash Equilibrium of the mining game.

:::{prf:definition} The Mining Game
:label: def-mining-game

The **Mining Game** $\Gamma$ is defined by:

- **Players:** $N$ miners indexed by $i \in \{1, \ldots, N\}$
- **Strategy Space:** Each miner chooses $\sigma_i \in \{\text{Honest}, \text{Cheat}\}$
  - **Honest:** Compute true gradient $g_i = \nabla_\Theta \mathcal{L}(\Theta, D)$
  - **Cheat:** Submit fake gradient $g_i' \neq g_{\text{true}}$
- **Payoffs:**
  - Block reward: $R > 0$ (received if block accepted)
  - Stake: $S > 0$ (lost if successfully challenged)
  - Computation cost: $C_{\text{honest}} > C_{\text{cheat}}$

**Key Assumptions:**
1. The detection probability $p_{\text{detect}}$ is exogenous (determined by the spot-check protocol, independent of other miners' strategies)
2. Block rewards are per-miner (not split among winners)
3. Miners play pure strategies (mixed strategies analyzed in Corollary {prf:ref}`cor-stake-reward-ratio`)

:::

:::{prf:theorem} The Verifier's Nash Equilibrium
:label: thm-verifier-nash-equilibrium

In the Mining Game $\Gamma$ with exogenous detection probability $p_{\text{detect}}$ and parameters satisfying:

$$
\frac{S}{R + S} > \frac{C_{\text{honest}} - C_{\text{cheat}}}{R}

$$

**Honest** is a strictly dominant strategy, and $\sigma^* = (\text{Honest}, \ldots, \text{Honest})$ is the unique Nash Equilibrium.

*Proof.*

**Step 1 (Utility Functions).** Define utilities for miner $i$:

$$
U_i(\text{Honest}) = R - C_{\text{honest}}

$$

$$
U_i(\text{Cheat}) = (1 - p_{\text{detect}}) \cdot R + p_{\text{detect}} \cdot (-S) - C_{\text{cheat}}

$$

where $p_{\text{detect}} \in (0, 1]$ is the probability of detection via spot-checking.

**Step 2 (Detection Probability).** By Theorem {prf:ref}`thm-holographic-verification`, detection probability satisfies:

$$
p_{\text{detect}} \geq 1 - (1 - \epsilon_{\min})^k

$$

for $k$ spot-check samples with $\epsilon_{\min} > 0$. Since $p_{\text{detect}}$ is exogenous (determined by the protocol, not other players), each miner faces a constant detection probability regardless of others' strategies.

**Step 3 (Incentive Compatibility).** Honesty is preferred when:

$$
U_i(\text{Honest}) > U_i(\text{Cheat})

$$

$$
R - C_{\text{honest}} > (1 - p_{\text{detect}}) R - p_{\text{detect}} S - C_{\text{cheat}}

$$

Rearranging:

$$
p_{\text{detect}} (R + S) > C_{\text{honest}} - C_{\text{cheat}}

$$

$$
p_{\text{detect}} > \frac{C_{\text{honest}} - C_{\text{cheat}}}{R + S} := p^*

$$

**Step 4 (Equilibrium Condition).** The theorem condition implies:

$$
\frac{S}{R + S} > \frac{C_{\text{honest}} - C_{\text{cheat}}}{R} \implies C_{\text{honest}} - C_{\text{cheat}} < \frac{S \cdot R}{R + S} < R

$$

Therefore $p^* = \frac{C_{\text{honest}} - C_{\text{cheat}}}{R + S} < 1$, ensuring the threshold is achievable with finite spot-checks.

**Step 5 (Dominant Strategy).** Since $p_{\text{detect}}$ is exogenous and independent of other players' strategies, miner $i$'s utility depends only on their own choice. When $p_{\text{detect}} > p^*$:

$$
\Delta U = U(\text{Cheat}) - U(\text{Honest}) = -p_{\text{detect}}(R + S) + (C_{\text{honest}} - C_{\text{cheat}}) < 0

$$

This holds regardless of what other miners do. Thus Honest is a **strictly dominant strategy**, and the unique Nash Equilibrium is all-Honest. $\square$

:::

:::{div} feynman-prose

Let me highlight what makes this theorem powerful: **strictly dominant strategy**.

In game theory, there are different kinds of equilibria. Some equilibria are fragile---they only work if everyone expects everyone else to play a certain way. If Alice thinks Bob might cheat, she might cheat too, and the whole thing unravels.

A strictly dominant strategy is different. It means: *no matter what anyone else does*, your best move is to play honestly. Even if every other miner in the network is cheating, *you* should still be honest. The incentives are personal, not collective.

This is crucial for a decentralized system. We cannot coordinate. We cannot trust each other. We cannot even verify who else is in the network. But we do not need to. Each miner, reasoning purely selfishly, concludes that honesty is their best bet. And when everyone reasons this way, everyone is honest.

The critical assumption is that detection probability is exogenous---it depends on the spot-check protocol, not on what other miners do. This is true in our scheme because challengers check against the *true* gradient (recomputed from the data), not against other miners' claims.

:::

:::{prf:corollary} The Stake-Reward Ratio
:label: cor-stake-reward-ratio

For the equilibrium to hold with detection probability $p_{\text{detect}} = 0.1$ (10% spot-check rate), the minimum stake-to-reward ratio is:

$$
\frac{S}{R} > \frac{C_{\text{honest}} - C_{\text{cheat}}}{p_{\text{detect}} \cdot R} - 1

$$

For typical gradient computation where $C_{\text{cheat}} = 0.1 C_{\text{honest}}$ (cheating saves 90% of compute), and assuming mining equilibrium where $R \approx C_{\text{honest}}$ (reward covers honest computation cost):

$$
\frac{S}{R} > \frac{0.9 C_{\text{honest}}}{0.1 \cdot R} - 1 = \frac{9 C_{\text{honest}}}{R} - 1 \approx 9 - 1 = 8

$$

*Interpretation:* Miners must stake approximately 8-10x the block reward to make cheating unprofitable with 10% spot-check rate.

:::

:::{div} feynman-prose

That 8-10x stake-to-reward ratio is actually quite reasonable. Here is what it means.

A miner who wants to earn 1 COG in block rewards must lock up approximately 8-10 COG as stake (depending on detection probability). If they cheat and get caught, they lose everything. If they are honest, they keep both the stake and the reward.

Now, here is the key insight: honest miners never lose their stake. It just sits there, block after block, earning rewards. The stake is not a cost; it is a security deposit. Over time, an honest miner earns many block rewards while their stake remains intact.

Compare this to Bitcoin, where miners must continuously spend on electricity---money that never comes back. In Proof of Useful Work, the stake is *recoverable*. You can unstake and leave whenever you want (after a cooldown period). The economic model is fundamentally different: capital commitment instead of ongoing consumption.

The 8-10x figure assumes a 10% spot-check rate and that cheating saves 90% of compute (a worst case). If the verification scheme is better---catching cheaters more often or earlier---the required stake can be lower. The corollary gives us a design equation: decide your detection probability, and the stake ratio follows.

:::

:::{admonition} Researcher Bridge: Connection to Optimistic Rollups
:class: info
:name: rb-optimistic-rollups

**Ethereum Optimistic Rollups:** Transactions are accepted optimistically; fraud proofs trigger rollback and slashing within a challenge window.

**PoUW Verification:** Gradients are accepted optimistically; spot-check challenges trigger slashing within a challenge window.

**Key Difference:** Optimistic Rollups verify **logic execution** (EVM traces). PoUW verifies **numerical computation** (gradient correctness). The Sieve provides the "virtual machine specification" against which fraudulent gradients are detected.

:::



(sec-consensus-minimum-friction-chain)=
## Consensus: The Minimum Friction Chain

:::{div} feynman-prose

Now we come to one of the most interesting parts: how does the network decide which chain is the "real" one when there are competing versions?

In Bitcoin, the rule is simple: follow the chain with the most work. If there is a fork, whichever branch has more hashes wins. This is the Nakamoto Consensus, and it works beautifully because hashes cannot be faked---the work proves itself.

But we are not doing hashes anymore; we are doing gradients. And gradients have a property that hashes do not: they are *meaningful*. A gradient that trains the model well is different from a gradient that damages it, even if both took the same computational effort to produce.

So here is the question: can we use this meaning to improve consensus? Can we pick not just the chain with the most work, but the chain with the *best* work?

The answer is yes, and the mechanism is beautiful. It is based on a concept called *metric friction*.

Think of each validator as having their own view of the world---their own internal model of what is going on. When validators train on the same data, their views converge. Their internal metrics become aligned, like iron filings lining up in a magnetic field. This is "gauge locking."

But when a validator submits a fraudulent gradient, their view diverges from everyone else's. There is friction between their metric and the honest validators' metrics. This friction is detectable.

The Minimum Friction Chain rule says: follow the chain that causes the least friction among validators. Honest chains lock gauges; fraudulent chains create friction. The network naturally selects for coherence.

:::

Nakamoto Consensus uses the "Heaviest Chain" rule. We introduce the **Minimum Friction Chain** rule derived from gauge theory.

:::{prf:definition} The Network Metric Tensor
:label: def-network-metric-tensor

Each validator $i$ maintains a local metric tensor $G^{(i)}$ on the shared latent manifold. The **Network Metric Friction** between chains $\mathcal{C}_A$ and $\mathcal{C}_B$ is:

$$
\mathcal{F}(\mathcal{C}_A, \mathcal{C}_B) := \sum_{i,j} \mathcal{F}_{ij}(\Theta_{\text{head}}^A, \Theta_{\text{head}}^B)

$$

where $\mathcal{F}_{ij}$ is the pairwise metric friction (Definition {prf:ref}`def-metric-friction`).

:::

:::{prf:definition} Metric Friction Consensus
:label: def-metric-friction-consensus

The **Canonical Chain** is selected by minimizing global metric friction:

$$
\mathcal{C}^* = \arg\min_{\mathcal{C}} \sum_{i < j} \mathcal{F}_{ij}(\Theta_{\text{head}}^\mathcal{C})

$$

*Mechanism:*
1. Miners propose competing updates $\{g_A, g_B, \ldots\}$
2. Validators compute local metric tensors $G^{(i)}(\Theta + g_k)$ for each candidate
3. The update minimizing pairwise friction is accepted
4. Ties broken by timestamp (first-seen)

:::

:::{prf:lemma} Gradient Observability
:label: lem-gradient-observability

The gradient $g$ uniquely determines the local metric tensor $G(\Theta + \epsilon g)$ to first order:

$$
G_{ij}(\Theta + \epsilon g) = G_{ij}(\Theta) + \epsilon \, \partial_k G_{ij} \cdot g^k + O(\epsilon^2)

$$

*Proof.* Direct Taylor expansion of the metric tensor. The metric is a smooth function of parameters, and its derivatives are observable from model predictions. $\square$

*Consequence:* Validators can infer each other's metrics from observed gradients without direct communication.

:::

:::{prf:theorem} Minimum Friction Byzantine Fault Tolerance
:label: thm-minimum-friction-bft

The Metric Friction Consensus achieves Byzantine Fault Tolerance against $f < N/3$ adversarial validators for **gradient-poisoning attacks** (adversaries submit incorrect gradients).

**Scope:** This theorem addresses data integrity attacks (model poisoning, fake gradients). Classical BFT attacks (equivocation, censorship) are handled by the underlying stake-based leader election, which is assumed to follow standard PBFT guarantees.

*Proof sketch.*

**Step 1 (Honest Majority Alignment).** By Theorem {prf:ref}`thm-spontaneous-gauge-locking`, honest validators minimizing prediction error on the same data undergo spontaneous gauge locking: $G^{(i)} \to G^{(j)}$ for honest $i, j$.

**Step 2 (Adversarial Inflation).** By Theorem {prf:ref}`thm-adversarial-mass-inflation` (Adversarial Mass Inflation), any gradient $g_{\text{adv}} \neq g_{\text{true}}$ introduces non-zero metric perturbation:

$$
\tilde{G}^{(i)} = G^{(i)} + \alpha_{\text{adv}} \mathcal{G}_{ij}, \quad \alpha_{\text{adv}} = \|g_{\text{adv}} - g_{\text{true}}\|_G > 0

$$

where $\mathcal{G}_{ij}$ is the Game Tensor (Definition {prf:ref}`def-gauge-covariant-game-tensor`). The key insight: *there is no "zero-curvature" way to submit a fake gradient*.

**Step 3 (Friction Separation).** Let $\epsilon$ be the natural gradient variance among honest validators. The pairwise friction satisfies:

- Honest-Honest: $\mathcal{F}_{ij} \leq c_1 \epsilon^2$ (gauge-locked, small noise)
- Honest-Adversarial: $\mathcal{F}_{ik} \geq c_2 \alpha_{\text{adv}}$ (metric mismatch)

For the attack to succeed while evading detection, the adversary requires $\alpha_{\text{adv}} < c_1 \epsilon^2 / c_2$. But such small perturbations have negligible effect on model training—a successful attack requires $\alpha_{\text{adv}} \gg \epsilon$.

**Step 4 (Selection).** The total friction of a chain proposed by honest validators is:

$$
\mathcal{F}_{\text{total}}^{\text{honest}} \leq \binom{N-f}{2} c_1 \epsilon^2 + f(N-f) c_2 \alpha_{\text{adv}}

$$

An adversarial chain has friction at least $\mathcal{F}_{\text{total}}^{\text{adv}} \geq (N-f) c_2 \alpha_{\text{adv}}$.

With $f < N/3$ and $\alpha_{\text{adv}} \gg \epsilon^2$, the honest chain minimizes total friction. $\square$

*Remark:* The $N/3$ threshold matches classical BFT because friction-weighted voting is equivalent to stake-weighted voting when adversarial friction is high.

:::

:::{prf:theorem} Adversarial Geometric Damping
:label: thm-adversarial-geometric-damping

An adversary controlling fraction $\alpha < 1/3$ of validators has influence on consensus bounded by:

$$
\|\Delta \Theta_{\text{adversarial}}\|_G \leq \frac{\alpha}{1 - 2\alpha} \|\Delta \Theta_{\text{honest}}\|_G

$$

*Proof.*

**Step 1.** The consensus update is a friction-weighted average:

$$
\Delta \Theta = \frac{\sum_i w_i \Delta \Theta^{(i)}}{\sum_i w_i}

$$

where weights $w_i = 1/\mathcal{F}_{i,\text{total}}$ penalize high-friction validators.

**Step 2.** Adversarial validators have inflated friction:

$$
w_{\text{adv}} \leq w_{\text{honest}} / (1 + \alpha_{\text{adv}}/\epsilon^2)

$$

**Step 3.** The adversarial contribution is:

$$
\|\Delta \Theta_{\text{adv}}\| \leq \frac{\alpha \cdot w_{\text{adv}}}{(1-\alpha) w_{\text{honest}} + \alpha w_{\text{adv}}} \|\Delta \Theta_{\text{total}}\|

$$

**Step 4.** Taking $w_{\text{adv}} \to 0$ in the limit of high adversarial friction:

$$
\|\Delta \Theta_{\text{adv}}\| \to 0

$$

The adversary is geometrically isolated. $\square$

*Interpretation:* Adversaries are not voted out---they are **geometrically damped**. Their updates carry infinite inertia (Causal Stasis) and cannot influence the consensus trajectory.

:::

:::{div} feynman-prose

I want to make sure you appreciate what just happened, because it is quite remarkable.

In traditional Byzantine Fault Tolerant systems, adversaries are dealt with by voting. If two-thirds of validators agree, the minority is outvoted. The honest majority forces the dishonest minority to comply.

But geometric damping works differently. We do not force adversaries to do anything. We do not vote them out. We do not even need to identify who they are. Instead, their updates simply *do not propagate*.

Here is the intuition. Every gradient update perturbs the metric tensor---it changes the shape of the space. Honest gradients perturb the metric in ways that are consistent with the data. Adversarial gradients perturb it in ways that are inconsistent.

When we weight updates by inverse friction, we are saying: updates that create a lot of geometric disruption get small weights. Updates that flow smoothly get large weights. The adversary's updates are geometrically disruptive by construction---they are trying to move the model somewhere the data does not support.

The result is that adversarial influence decays exponentially with the amount of geometric mismatch. The adversary is not defeated by force; they are defeated by irrelevance. Their voice becomes static, their influence becomes noise, and the honest trajectory continues unperturbed.

This is consensus through coherence, not consensus through coercion.

:::



(sec-tokenomics-thermodynamic-value)=
## Tokenomics: Thermodynamic Value

:::{div} feynman-prose

Now we come to the economics, and this is where things get philosophically interesting.

What is money? Usually we think of it as a social agreement---a piece of paper that we all *pretend* has value. Fiat currency has no intrinsic worth; its value comes from trust in the government that issues it.

Bitcoin tried to change this by making money scarce. There will only ever be 21 million bitcoins, and producing each one requires real physical work. But that work is *arbitrary*---the hash computation serves no purpose except to be difficult.

The COG token takes the next step. Each token is not just scarce; it represents *actual cognitive work* done on behalf of the network. When you hold a COG token, you hold a certificate that says: "Someone spent energy teaching the global model. This energy is permanently recorded in the parameters of a neural network that now knows more than it did before."

This is a profound shift in what "backing" a currency means. The COG is not backed by gold, or by government decree, or by scarcity alone. It is backed by *knowledge*. Every token minted represents a real reduction in the model's uncertainty about the world.

And here is the beautiful part: as the model gets smarter, its output gets more valuable. The token is tied to something that actually improves over time.

:::

The native token ($\text{COG}$) is not fiat currency---it is a **thermodynamic certificate** representing stored cognitive work.

:::{prf:definition} The Token Standard
:label: def-token-standard

The $\text{COG}$ token has three fundamental operations:

1. **Minting (Supply).** Tokens are minted when **Ontological Stress** $\Xi$ is reduced:

$$
\Delta \text{Supply} = \kappa_{\text{mint}} \cdot \max(0, -\Delta \Xi_{\text{global}})

$$

where $\kappa_{\text{mint}}$ is the minting coefficient (tokens per nat of stress reduction).

*Interpretation:* Value is created only when the network learns something new.

2. **Burning (Demand).** Tokens are burned to request **Inference**:

$$
\text{Cost}(Q) = \mathfrak{T}_{\text{harvest}}^{-1}(\dot{\mathcal{M}}_Q)

$$

where $\dot{\mathcal{M}}_Q$ is the metabolic cost of answering query $Q$.

3. **Transfer.** Standard ERC-20-like transfers between accounts.

*Units:* $[\text{COG}] = \text{Joules}$ (energy equivalent).

*Value Anchor:* $1 \, \text{COG} \approx 1 \, \text{Joule}$ of useful gradient computation at reference temperature $T_c$.

:::

:::{prf:theorem} Value-Intelligence Coupling
:label: thm-value-intelligence-coupling

The equilibrium token price $P_{\text{COG}}$ is bounded by:

$$
P_{\text{floor}} \leq P_{\text{COG}} \leq P_{\text{ceiling}}

$$

where:

$$
P_{\text{floor}} = C_{\text{electricity}} \cdot J_{\text{per\_COG}}

$$

(cost of electricity to generate one COG worth of computation)

$$
P_{\text{ceiling}} = \frac{V_{\text{inference}}}{J_{\text{per\_query}}}

$$

(value of inference output per Joule)

*Proof.*

**Step 1 (Floor).** If $P_{\text{COG}} < P_{\text{floor}}$, miners cannot profitably produce blocks. Supply decreases until price rises.

**Step 2 (Ceiling).** If $P_{\text{COG}} > P_{\text{ceiling}}$, users won't pay for inference. Demand decreases until price falls.

**Step 3 (Equilibrium).** At equilibrium:

$$
P_{\text{COG}}^* = \sqrt{P_{\text{floor}} \cdot P_{\text{ceiling}}}

$$

(geometric mean under log-linear supply/demand). $\square$

:::

:::{prf:corollary} Intelligence-Price Feedback
:label: cor-intelligence-price-feedback

As the model improves:

1. Inference value $V_{\text{inference}} \uparrow$
2. Ceiling $P_{\text{ceiling}} \uparrow$
3. Equilibrium price $P_{\text{COG}}^* \uparrow$
4. Mining profitability $\uparrow$
5. More compute allocated $\uparrow$
6. Model improves faster $\uparrow$

This creates a **positive feedback loop** between intelligence and economic value.

:::

:::{div} feynman-prose

Let me walk you through this feedback loop, because it is the engine that drives the whole system.

**The virtuous cycle:**

1. People want to use the model for inference (answering questions, making predictions, generating content).
2. They pay for inference in COG tokens. These tokens are burned.
3. Burning tokens reduces supply, which increases price (if demand is constant).
4. Higher price means mining is more profitable.
5. More miners join, contributing more compute.
6. More compute means faster training, which improves the model.
7. Better model means more valuable inference.
8. Back to step 1.

This is not just a speculative bubble. The price increase is tied to *real capability improvement*. As the model gets smarter, it can do more valuable things: answer harder questions, write better code, make more accurate predictions. The token price tracks the model's competence.

Compare this to Bitcoin, where the feedback loop is purely financial. More demand raises the price, which attracts more miners, which increases security, which increases trust, which increases demand. There is no improvement in the underlying asset---Bitcoin in 2035 will do exactly what Bitcoin did in 2015, just more securely.

In Proof of Useful Work, the asset itself gets better. The network is not just maintaining value; it is creating it.

**The floor and ceiling:**

The equilibrium price is bounded by two things:
- **Floor:** The cost of electricity to produce one COG worth of computation. Below this price, miners lose money and shut down.
- **Ceiling:** The value of inference output per unit cost. Above this price, users switch to alternatives.

Between these bounds, the price floats freely based on supply and demand. But the bounds themselves move: as hardware improves, the floor drops; as the model improves, the ceiling rises. The long-term trend is for both to increase, with the price tracking somewhere in between.

:::

:::{admonition} Physics Isomorphism: The Token as Gibbs Free Energy
:class: note
:name: pi-gibbs-free-energy

**In Physics:** Gibbs Free Energy $G = H - TS$ measures the maximum useful work extractable from a system at constant temperature and pressure.

**In Implementation:** The COG token measures the maximum useful computation extractable from the network:

$$
\text{Value}(\text{COG}) = \mathfrak{T}_{\text{harvest}}(r) - T_c \cdot S_{\text{overhead}}

$$

where $S_{\text{overhead}}$ is the entropy cost of coordination.

**Correspondence Table:**

| Thermodynamics | Token Economics |
|:---------------|:----------------|
| Gibbs Free Energy $G$ | Token Value |
| Enthalpy $H$ | Gross computation |
| Entropy $S$ | Coordination overhead |
| Temperature $T$ | Cognitive temperature $T_c$ |
| Work extraction | Inference service |

:::



(sec-the-ledger-as-holographic-screen)=
## The Ledger as Holographic Screen

:::{div} feynman-prose

Now we come to what I think is the deepest idea in this whole section, and it connects to something physicists have been puzzling over since Bekenstein and Hawking studied black holes.

The holographic principle says: the maximum information that can be stored in a region of space is proportional to its *surface area*, not its volume. This is bizarre. You would think a bigger box could hold more information. But no---what matters is the boundary.

Black holes make this concrete. The entropy of a black hole is proportional to the area of its event horizon. All the information about whatever fell in is somehow encoded on that two-dimensional surface.

Now here is the connection. The blockchain is a *boundary* too. It is the interface between the past (what has been learned) and the future (what remains to learn). Every block adds a layer to this boundary, recording the gradients that shaped the model.

And just like a holographic screen in physics, the blockchain encodes the bulk. Given the chain, you can reconstruct the model---run through all the gradients starting from random initialization, and you get the current weights. The three-dimensional "bulk" (the full model) is determined by the one-dimensional "boundary" (the sequence of blocks).

This is not just a metaphor. The information bounds work out. The Causal Information Bound (which we used for verification) is essentially a holographic bound: bulk information cannot exceed boundary capacity. The blockchain respects this bound by construction.

:::

We prove that the blockchain is the discrete realization of the **Memory Screen** (Definition {prf:ref}`def-memory-screen`).

:::{prf:theorem} Ledger-Memory Screen Isomorphism
:label: thm-ledger-memory-isomorphism

Let $\Xi_T$ be the Memory Screen (Definition {prf:ref}`def-memory-screen`) and $\mathcal{L}_H$ be the blockchain of height $H$. There exists an isomorphism:

$$
\Phi: \mathcal{L}_H \to \Xi_T

$$

given by:

| Blockchain | Memory Screen | Symbol |
|:-----------|:--------------|:-------|
| Block height $h$ | Time coordinate $t$ | $h \leftrightarrow t$ |
| Merkle root $\mathcal{H}_h$ | Boundary state $z_{\partial}$ | $\mathcal{H}_h \leftrightarrow z_{\partial}(t)$ |
| Gradient $g_h$ | Flux $\alpha(t)$ | $g_h \leftrightarrow \alpha(t)$ |
| Chain $\sum_{h=0}^H B_h$ | Screen $\int_0^T \alpha(t) \delta_{\gamma(t)} dt$ | $\mathcal{L}_H \leftrightarrow \Xi_T$ |

*Proof.*

**Step 1.** The Memory Screen (Definition {prf:ref}`def-memory-screen`) is:

$$
\Xi_T = \int_0^T \alpha(t') \, \delta_{\gamma(t')} \, dt'

$$

where $\alpha(t) = J_r(t)$ is the reward flux and $\gamma(t)$ is the trajectory.

**Step 2.** The blockchain is:

$$
\mathcal{L}_H = \sum_{h=0}^{H} B_h = \sum_{h=0}^{H} (g_h, \mathcal{H}_h, \ldots)

$$

**Step 3.** Define the correspondence:
- $t = h \cdot \Delta t$ where $\Delta t$ is block time
- $\alpha(t) = g_h / \Delta t$ (gradient rate)
- $\gamma(h) = \Theta_h$ (parameter trajectory)

**Step 4.** The discrete sum converges to the continuous integral:

$$
\sum_{h=0}^{H} g_h \cdot \mathbb{1}_{\Theta_h} \to \int_0^T \alpha(t) \delta_{\gamma(t)} dt

$$

as $\Delta t \to 0$. $\square$

*Interpretation:* The blockchain is the **frozen boundary** of the network's cognitive trajectory. Each block records a moment of learning; the full chain is the holographic screen encoding the network's history.

:::

:::{prf:corollary} Block Size from Area Law
:label: cor-block-size-area-law

The maximum information in a block is bounded by:

$$
I_{\text{block}} \leq \nu_D \cdot \frac{\text{Area}(\partial \mathcal{Z})}{\ell_L^{D-1}}

$$

where the area is measured in the header's Merkle tree.

*Proof.* Direct application of Theorem {prf:ref}`thm-causal-information-bound` to the block's boundary. $\square$

*Consequence:* Oversized blocks violate the Causal Information Bound. The network enters **Causal Stasis** (Theorem {prf:ref}`thm-causal-stasis`) if blocks exceed capacity—propagation delay exceeds block time.

:::

:::{prf:definition} Chain Renormalization (Pruning)
:label: def-chain-renormalization

Old blocks are **coarse-grained** into **Epoch Blocks** via the Projection Operator:

$$
B_{\text{epoch}} = \Pi\left( \sum_{h \in \text{epoch}} B_h \right)

$$

where $\Pi$ projects onto the low-frequency components of the gradient history.

*Mechanism:*
1. Every $N_{\text{epoch}}$ blocks, compress the epoch into a summary
2. Discard individual block data (retain Merkle proofs)
3. The agent remembers the "gist" but forgets the "noise"

*Thermodynamics:* This is **information erasure** (Landauer cost). It releases storage but maintains the essential learning trajectory.

:::

:::{div} feynman-prose

The chain renormalization is worth thinking about carefully, because it tells us something about the nature of memory itself.

As the blockchain grows, it becomes unwieldy. Thousands, then millions, then billions of blocks---each containing gradients, certificates, metadata. At some point, you cannot keep all of it. You must *forget*.

But forgetting is not free. The Landauer principle says: erasing information costs energy. When you compress old blocks into epoch summaries, you are literally paying to forget.

However---and this is the key---not all information is equally important. Early in training, the model makes big, dramatic updates. Later, it makes small refinements. The early gradients carry more "weight" in a sense---they determined the broad structure of what the model knows.

The projection operator $\Pi$ captures this. It keeps the low-frequency components---the big picture, the major trends---and discards the high-frequency noise. This is exactly like human memory. You remember the gist of what happened years ago, but not the exact words of every conversation. The fine details fade; the structure remains.

The result is a hierarchical memory: recent blocks in full detail, older epochs in summary form, ancient history in broad strokes. The model remembers its past, but not perfectly. This is not a bug; it is a feature. Perfect memory would be infinitely expensive. Selective memory is efficient and sufficient.

:::



(sec-security-analysis)=
## Security Analysis

:::{div} feynman-prose

Now we need to kick the tires on this system. Security analysis is where we ask: what happens when someone *tries* to break it?

Every blockchain faces certain canonical attacks. The 51% attack: what if an adversary controls most of the compute? Flash loan attacks: what if someone manipulates prices faster than the network can respond? Corruption: what if validators collude to deceive everyone else?

What makes Proof of Useful Work interesting is that these attacks have *geometric* consequences. When you try to rewrite history, you are not just fighting an economic battle; you are fighting a geometric one. You have to produce gradients that are consistent with data that does not exist, which creates friction, which makes your chain detectably anomalous.

Let us go through the major attacks and see how they are handled.

:::

We analyze resistance to standard blockchain attacks through the geometric lens.

:::{prf:theorem} 51% Attack Geometric Rejection
:label: thm-51-attack-rejection

An attacker controlling $> 50\%$ of compute cannot rewrite history without triggering **Spontaneous Fission**.

*Proof.*

**Step 1.** The attacker proposes an alternative chain $\mathcal{C}'$ that contradicts the Memory Screen $\Xi_T$ of honest validators.

**Step 2.** By Theorem {prf:ref}`thm-adversarial-mass-inflation`, the attacker's chain has inflated metric:

$$
\tilde{G}_{\text{attack}} = G + \alpha_{\text{adv}} \mathcal{G}

$$

**Step 3.** The Metric Friction between honest and attack chains is:

$$
\mathcal{F}(\mathcal{C}, \mathcal{C}') = \|G - \tilde{G}_{\text{attack}}\|_F^2 \sim O(\alpha_{\text{adv}}^2)

$$

**Step 4.** When $\mathcal{F} > \mathcal{F}_{\text{crit}}$ (Fission Threshold from Theorem {prf:ref}`thm-fission-criterion`), the network undergoes **Spontaneous Fission**:
- The attacker ends up on a high-friction shard
- The honest validators continue on the low-friction chain

**Step 5.** The attacker's shard enters **Causal Stasis** (Theorem {prf:ref}`thm-causal-stasis`)---no one provides data/compute, and it dies. $\square$

*Interpretation:* You cannot buy the network because you cannot buy **geometric alignment**.

:::

:::{div} feynman-prose

The 51% attack is the canonical boogeyman of blockchain security. In Bitcoin, if you control 51% of the hashrate, you can rewrite history: mine a secret chain, wait until it is longer than the public one, then reveal it and invalidate all the transactions you want to undo.

In Proof of Useful Work, this attack *does not work the same way*. Here is why.

When an attacker tries to create an alternative history, they must produce gradients that explain their version of events. But the honest validators have their own memory---they remember what the model looked like at each block. The attacker's chain will have different gradients, which means it will lead to a different model state.

Now, here is the key: the metric friction between the attacker's model and the honest validators' models is high. The attacker is claiming a different reality, and that difference shows up geometrically. When the network compares chains using the Minimum Friction rule, the attacker's chain creates more friction.

If the friction is high enough, the network *fissions*. The attacker ends up on their own shard, talking to themselves. No honest validator will follow them because following them increases friction. The attacker has not "won"---they have exiled themselves.

And then the final blow: the attacker's shard enters Causal Stasis. No one submits new training data to them. No one queries their model for inference. They have 51% of the compute, but 0% of the economic activity. Their chain dies of neglect.

This is security through coherence. You cannot buy alignment; you can only earn it by being honest.

:::

:::{prf:theorem} Causal Theft Prevention
:label: thm-causal-theft-prevention

Flash-loan attacks and front-running are rejected by **CausalityViolationCheck (Node 62)**.

*Proof.*

**Step 1.** A flash-loan attack requires: borrow $\to$ manipulate price $\to$ profit $\to$ repay, all in one transaction.

**Step 2.** The profit depends on a price change that **hasn't propagated** in the causal graph at the time of the borrow.

**Step 3.** By the Causal Information Bound (Theorem {prf:ref}`thm-causal-information-bound`), information cannot propagate faster than:

$$
v_{\max} = \frac{d_G(z, z')}{t}

$$

**Step 4.** **Node 62 (CausalityViolationCheck)** detects transactions using information from the future:

$$
\Delta_{\text{causal}} = D_{\text{KL}}(P_{\text{interventional}} \| P_{\text{observational}}) > \delta_{\text{causal}}

$$

**Step 5.** The transaction is rejected as **geometrically impossible**. $\square$

:::

:::{div} feynman-prose

Flash loans are a fascinating attack vector that emerged in DeFi. The idea is: borrow a huge amount of money, use it to manipulate a market, profit from the manipulation, repay the loan---all in a single transaction. You need no capital of your own; you just need to find a sequence of operations that nets positive.

These attacks exploit the fact that information in a blockchain propagates slowly. You can see a price update coming and get ahead of it (front-running), or you can create a price change and exploit it before anyone can react.

The Causal Information Bound provides a deep defense. Here is the intuition.

Every piece of information in the system has a *causal history*---a chain of events that produced it. When you look at a price, that price came from trades, which came from decisions, which came from observations. This chain has a finite propagation speed.

A flash loan attack tries to use information from the *future*---the price after the manipulation---to guide the action *before* the manipulation. This is causal theft. You are acting on information that you should not have yet.

Node 62 (CausalityViolationCheck) detects this by comparing two probability distributions:
- $P_{\text{observational}}$: what you *should* have known at the time you acted
- $P_{\text{interventional}}$: what you *did* know, as revealed by your actions

If the gap is too large---if your actions reveal knowledge of the future---the transaction is rejected. Not because we caught you cheating, but because your behavior is *geometrically inconsistent* with the causal structure of reality.

:::

:::{prf:theorem} Corruption Detection via Babel Limit
:label: thm-corruption-babel-detection

Sustained deception by corrupt actors exceeds the **Babel Limit** (Theorem {prf:ref}`thm-babel-limit`) and causes loss of gauge locking.

*Proof.*

**Step 1.** A corrupt actor broadcasts metric $G_{\text{corrupt}}$ claiming to optimize the objective, but their actual gradient flow generates different geometry.

**Step 2.** Maintaining the deception requires transmitting additional fake metric information:

$$
I_{\text{deception}} = H(G_{\text{corrupt}}) - H(G_{\text{true}})

$$

**Step 3.** By Theorem {prf:ref}`thm-babel-limit`, complete gauge locking requires:

$$
\dim(\mathfrak{g}) \cdot H(G) \leq C_{\mathcal{L}}

$$

**Step 4.** The deception increases effective entropy, violating the Babel Limit:

$$
\dim(\mathfrak{g}) \cdot (H(G_{\text{true}}) + I_{\text{deception}}) > C_{\mathcal{L}}

$$

**Step 5.** The corrupt actor loses gauge locking with honest validators. Their words become "noise"---they are **topologically exiled** from consensus. $\square$

*Interpretation:* You cannot lie to the network because you cannot fake the **thermodynamic trace** of your actions.

:::

:::{div} feynman-prose

The Babel Limit attack is perhaps the most insidious: what if validators *collude* to deceive everyone? They coordinate in secret, they agree to broadcast false information, they maintain a consistent story. How can the network detect this?

The answer comes from information theory, specifically from the Babel Limit theorem that we proved earlier in the book. Here is the essence.

To maintain a deception, the corrupt actors must coordinate. They need to agree on the lie and keep their stories straight. This coordination requires *communication*. They must exchange enough information to synchronize their fake metrics.

But communication has limits. The channel capacity between any two validators is finite. The amount of coordination they can achieve is bounded by this capacity.

Meanwhile, the honest validators are all learning from the same data. Their metrics *naturally* converge, without needing to communicate much at all. The data itself provides the coordination signal.

The corrupt actors face a dilemma:
- If their lie is simple, it creates detectable metric friction with honest validators.
- If their lie is complex (to avoid friction), they need more coordination than their channel capacity allows.

The Babel Limit is the information-theoretic boundary where deception becomes impossible. Below it, you can maintain a consistent lie for a while. Above it, your lie becomes inconsistent, your metrics diverge, and you lose gauge locking.

And once you lose gauge locking, your messages become unintelligible to honest validators. You are not refuted; you are *rendered meaningless*. Your words are noise. You have exiled yourself from consensus not by being wrong, but by being incoherent.

This is, I think, a genuinely new kind of security property. Traditional security says: we can detect your lie. Babel security says: we do not need to detect your lie, because you cannot tell it coherently in the first place.

:::



(sec-implementation-cognichain)=
## Implementation: The CogniChain Module

We provide a reference implementation combining the Sieve, Metabolic Battery, and Consensus mechanisms.

```python
"""
CogniChain: Proof of Useful Work Implementation

Reference implementation for Section 38 (Proof of Useful Work).
Combines gradient mining with Sieve validation and thermodynamic verification.

Cross-references:
    - Metabolic Transducer: Definition `def-metabolic-transducer`
    - Causal Information Bound: Theorem `thm-causal-information-bound`
    - Verifier's Nash Equilibrium: Theorem `thm-verifier-nash-equilibrium`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import hashlib


@dataclass
class BlockConfig:
    """Configuration for Curriculum Blocks.

    Units:
        stake_amount: [COG] (token units)
        batch_size_min: [samples]
        challenge_window: [seconds]
    """
    stake_amount: float = 1000.0
    batch_size_min: int = 1024
    challenge_window: float = 600.0  # 10 minutes
    spot_check_fraction: float = 0.01
    cosine_threshold: float = 0.9
    gradient_norm_max: float = 100.0


@dataclass
class SieveCertificate:
    """Sieve validation certificate (Definition `def-boundary-flux-certificate`).

    Contains boundary flux data for holographic verification.
    """
    gradient_norm: float
    boundary_flux: torch.Tensor
    hessian_trace: float
    sample_seed: int
    is_valid: bool
    violations: List[str] = field(default_factory=list)


@dataclass
class CurriculumBlock:
    """A block in the Cognitive Ledger (Definition `def-curriculum-block`).

    Attributes:
        prev_hash: Hash of previous block
        data_hash: Content identifier of training batch (IPFS CID)
        gradient: The computed gradient (the "Work")
        stake_proof: Signature proving stake ownership
        sieve_cert: Validation certificate from Sieve
        timestamp: Block creation time
    """
    prev_hash: str
    data_hash: str
    gradient: torch.Tensor
    stake_proof: str
    sieve_cert: SieveCertificate
    timestamp: float = 0.0

    def compute_hash(self) -> str:
        """Compute block hash for chain linking."""
        data = f"{self.prev_hash}{self.data_hash}{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()


class GradientSieve(nn.Module):
    """
    Sieve validation for gradient mining.

    Implements Nodes 1, 29, 53 checks for gradient validity.
    Returns a SieveCertificate for holographic verification.

    References:
        - CostBoundCheck (Node 1): {ref}`sec-the-stability-checks`
        - TextureFirewallCheck (Node 29): {ref}`sec-motor-texture-the-action-residual`
        - CausalEnclosureCheck (Node 53): {ref}`sec-implementation-the-experimental-sieve`
    """

    def __init__(self, config: BlockConfig):
        super().__init__()
        self.config = config

    def compute_gradient_norm(
        self,
        gradient: torch.Tensor,
        metric: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute gradient norm in capacity-constrained metric.

        Args:
            gradient: [D] gradient vector
            metric: [D, D] metric tensor (default: identity)

        Returns:
            ||g||_G = sqrt(g^T G g)
        """
        if metric is None:
            return gradient.norm().item()
        return torch.sqrt(gradient @ metric @ gradient).item()

    def check_cost_bound(self, gradient: torch.Tensor) -> Tuple[bool, str]:
        """
        Node 1: CostBoundCheck - Gradient energy must be bounded.

        Args:
            gradient: [D] gradient vector

        Returns:
            (is_valid, message)
        """
        norm = self.compute_gradient_norm(gradient)
        if norm > self.config.gradient_norm_max:
            return False, f"Gradient norm {norm:.2f} exceeds max {self.config.gradient_norm_max}"
        if not torch.isfinite(gradient).all():
            return False, "Gradient contains non-finite values"
        return True, "CostBoundCheck passed"

    def check_texture_firewall(
        self,
        gradient: torch.Tensor,
        texture_indices: Optional[List[int]] = None
    ) -> Tuple[bool, str]:
        """
        Node 29: TextureFirewallCheck - No texture gradient leakage.

        Args:
            gradient: [D] gradient vector
            texture_indices: Indices of texture coordinates

        Returns:
            (is_valid, message)
        """
        if texture_indices is None:
            # Default: last 10% of dimensions are texture
            d = len(gradient)
            texture_indices = list(range(int(0.9 * d), d))

        texture_grad = gradient[texture_indices]
        texture_norm = texture_grad.norm().item()

        # Texture gradient should be near-zero
        eps_texture = 1e-3 * gradient.norm().item()
        if texture_norm > eps_texture:
            return False, f"Texture gradient {texture_norm:.2e} exceeds threshold {eps_texture:.2e}"
        return True, "TextureFirewallCheck passed"

    def check_causal_enclosure(
        self,
        gradient: torch.Tensor,
        model: nn.Module,
        data: torch.Tensor
    ) -> Tuple[bool, str]:
        """
        Node 53: CausalEnclosureCheck - Interventional gap is bounded.

        Args:
            gradient: [D] claimed gradient
            model: Model to verify against
            data: Training data batch

        Returns:
            (is_valid, message)
        """
        # Compute true gradient on a subset
        subset_size = max(1, int(len(data) * self.config.spot_check_fraction))
        subset_idx = torch.randperm(len(data))[:subset_size]
        subset_data = data[subset_idx]

        model.zero_grad()
        loss = model(subset_data).mean()  # Simplified loss
        loss.backward()

        true_grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

        # Normalize gradients for comparison
        grad_norm = gradient[:len(true_grad)]
        if grad_norm.norm() > 0 and true_grad.norm() > 0:
            cosine_sim = F.cosine_similarity(
                grad_norm.unsqueeze(0),
                true_grad.unsqueeze(0)
            ).item()
        else:
            cosine_sim = 0.0

        delta_causal = 1.0 - cosine_sim

        if delta_causal > (1.0 - self.config.cosine_threshold):
            return False, f"Causal gap {delta_causal:.3f} exceeds threshold"
        return True, f"CausalEnclosureCheck passed (cosine={cosine_sim:.3f})"

    def validate(
        self,
        gradient: torch.Tensor,
        model: Optional[nn.Module] = None,
        data: Optional[torch.Tensor] = None
    ) -> SieveCertificate:
        """
        Run full Sieve validation on a gradient.

        Args:
            gradient: [D] gradient vector to validate
            model: Model for causal check (optional)
            data: Data batch for causal check (optional)

        Returns:
            SieveCertificate with validation results
        """
        violations = []
        is_valid = True

        # Node 1: Cost Bound
        valid_1, msg_1 = self.check_cost_bound(gradient)
        if not valid_1:
            is_valid = False
            violations.append(msg_1)

        # Node 29: Texture Firewall
        valid_29, msg_29 = self.check_texture_firewall(gradient)
        if not valid_29:
            is_valid = False
            violations.append(msg_29)

        # Node 53: Causal Enclosure (if model/data provided)
        if model is not None and data is not None:
            valid_53, msg_53 = self.check_causal_enclosure(gradient, model, data)
            if not valid_53:
                is_valid = False
                violations.append(msg_53)

        # Compute certificate data
        grad_norm = self.compute_gradient_norm(gradient)

        # Boundary flux: simplified as first few gradient components
        boundary_dim = min(16, len(gradient))
        boundary_flux = gradient[:boundary_dim].clone()

        # Hessian trace: approximate via finite differences (stub)
        hessian_trace = grad_norm  # Simplified proxy

        return SieveCertificate(
            gradient_norm=grad_norm,
            boundary_flux=boundary_flux,
            hessian_trace=hessian_trace,
            sample_seed=torch.randint(0, 2**31, (1,)).item(),
            is_valid=is_valid,
            violations=violations
        )


class CogniChainNode(nn.Module):
    """
    A node in the Proof-of-Useful-Work network.

    Combines the Metabolic Transducer, Sieve validation, and consensus logic.

    References:
        - Gradient Mining: Definition `def-gradient-mining-puzzle`
        - Holographic Verification: Theorem `thm-holographic-verification`
        - Nash Equilibrium: Theorem `thm-verifier-nash-equilibrium`
    """

    def __init__(
        self,
        model: nn.Module,
        config: BlockConfig,
        stake: float = 1000.0
    ):
        """
        Args:
            model: The global model to train
            config: Block configuration
            stake: Amount of COG tokens staked
        """
        super().__init__()
        self.model = model
        self.config = config
        self.sieve = GradientSieve(config)
        self.stake = stake
        self.chain: List[CurriculumBlock] = []
        self.model_params = sum(p.numel() for p in model.parameters())

    def mine_block(
        self,
        data_batch: torch.Tensor,
        loss_fn: nn.Module
    ) -> CurriculumBlock:
        """
        Mine a new block via gradient computation.

        This is the "Mining" process: SGD + Sieve Validation.
        Replaces SHA-256 hashing with Gradient Computation.

        Args:
            data_batch: [B, ...] Training data batch
            loss_fn: Loss function module

        Returns:
            CurriculumBlock if valid, raises ValueError if Sieve violated

        Raises:
            ValueError: If gradient fails Sieve validation
        """
        # 1. Validate batch size meets difficulty
        if len(data_batch) < self.config.batch_size_min:
            raise ValueError(
                f"Batch size {len(data_batch)} < minimum {self.config.batch_size_min}"
            )

        # 2. Compute Gradient (The Work)
        self.model.zero_grad()
        outputs = self.model(data_batch)
        loss = loss_fn(outputs, data_batch)
        loss.backward()

        gradient = torch.cat([
            p.grad.flatten()
            for p in self.model.parameters()
            if p.grad is not None
        ])

        # 3. Run Sieve Validation
        cert = self.sieve.validate(gradient, self.model, data_batch)

        if not cert.is_valid:
            raise ValueError(f"Mining failed: {cert.violations}")

        # 4. Create Block
        prev_hash = self.chain[-1].compute_hash() if self.chain else "0" * 64
        data_hash = hashlib.sha256(data_batch.cpu().numpy().tobytes()).hexdigest()

        block = CurriculumBlock(
            prev_hash=prev_hash,
            data_hash=data_hash,
            gradient=gradient.detach(),
            stake_proof=self._sign_stake(),
            sieve_cert=cert,
            timestamp=0.0  # Would use real timestamp
        )

        return block

    def verify_block(
        self,
        block: CurriculumBlock,
        sample_data: torch.Tensor
    ) -> Tuple[bool, str]:
        """
        Verify a block using Holographic Verification.

        Implements the Optimistic Verification Protocol with spot-checking.

        Args:
            block: Block to verify
            sample_data: Random sample of training data for spot-check

        Returns:
            (is_valid, message)
        """
        # 1. Check Stake Proof
        if not self._verify_stake(block.stake_proof):
            return False, "Invalid stake proof"

        # 2. Check Sieve Certificate (fast path)
        if not block.sieve_cert.is_valid:
            return False, f"Sieve violations: {block.sieve_cert.violations}"

        # 3. Energy Conservation Check (Holographic)
        # Gradient norm must be consistent with claimed computation
        if block.sieve_cert.gradient_norm > self.config.gradient_norm_max:
            return False, "Gradient norm exceeds capacity bound"

        # 4. Spot-Check Gradient Direction
        self.model.zero_grad()
        outputs = self.model(sample_data)
        loss = outputs.mean()  # Simplified
        loss.backward()

        my_grad = torch.cat([
            p.grad.flatten()
            for p in self.model.parameters()
            if p.grad is not None
        ])

        # Compare direction via cosine similarity
        block_grad = block.gradient[:len(my_grad)]
        if block_grad.norm() > 0 and my_grad.norm() > 0:
            cosine_sim = F.cosine_similarity(
                block_grad.unsqueeze(0),
                my_grad.unsqueeze(0)
            ).item()
        else:
            cosine_sim = 0.0

        if cosine_sim < self.config.cosine_threshold:
            return False, f"Gradient direction mismatch: cosine={cosine_sim:.3f}"

        return True, f"Block verified (cosine={cosine_sim:.3f})"

    def apply_block(self, block: CurriculumBlock, learning_rate: float = 0.01):
        """
        Apply a verified block to update the model.

        Implements the Chain Evolution Rule (Definition `def-chain-evolution`).

        Args:
            block: Verified block to apply
            learning_rate: Learning rate eta_h
        """
        # Reconstruct gradient for each parameter
        idx = 0
        with torch.no_grad():
            for param in self.model.parameters():
                numel = param.numel()
                param_grad = block.gradient[idx:idx + numel].view(param.shape)
                param.sub_(learning_rate * param_grad)
                idx += numel

        self.chain.append(block)

    def compute_metric_friction(
        self,
        other_gradient: torch.Tensor
    ) -> float:
        """
        Compute metric friction with another validator's gradient.

        Implements Definition `def-metric-friction` for consensus.

        Args:
            other_gradient: [D] gradient from another validator

        Returns:
            Metric friction F_AB (dimensionless)
        """
        if len(self.chain) == 0:
            return 0.0

        my_grad = self.chain[-1].gradient
        min_len = min(len(my_grad), len(other_gradient))

        # Frobenius norm of difference (simplified metric friction)
        friction = F.mse_loss(
            my_grad[:min_len],
            other_gradient[:min_len]
        ).item()

        return friction

    def _sign_stake(self) -> str:
        """Generate stake proof (placeholder)."""
        return f"STAKE_PROOF_{self.stake}"

    def _verify_stake(self, proof: str) -> bool:
        """Verify stake proof (placeholder)."""
        return proof.startswith("STAKE_PROOF_")


def compute_network_consensus(
    validators: List[CogniChainNode],
    candidate_blocks: List[CurriculumBlock]
) -> int:
    """
    Select winning block via Minimum Friction Consensus.

    Implements Definition `def-metric-friction-consensus`.

    Args:
        validators: List of validator nodes
        candidate_blocks: Competing blocks for this height

    Returns:
        Index of winning block
    """
    if len(candidate_blocks) == 0:
        raise ValueError("No candidate blocks")
    if len(candidate_blocks) == 1:
        return 0

    # Compute total friction for each candidate
    frictions = []
    for block in candidate_blocks:
        total_friction = 0.0
        for v1 in validators:
            for v2 in validators:
                if v1 is not v2:
                    f = v1.compute_metric_friction(block.gradient)
                    total_friction += f
        frictions.append(total_friction)

    # Select minimum friction
    return int(torch.tensor(frictions).argmin().item())
```



(sec-summary-proof-useful-work)=
## Summary

:::{div} feynman-prose

Let me step back and reflect on what we have built.

We started with a simple observation: Bitcoin wastes energy. The security comes from doing hard work, but the work itself produces nothing of lasting value. Trillions of hashes computed, and at the end of it, all we have is a number below a target and a lot of heat.

We asked: can we do better? Can we have the security without the waste?

The answer, it turns out, is yes. The key insight is that the Landauer bound does not care *what* you compute---it only cares how many bits you erase. Computing gradients erases as many bits as computing hashes. So we can swap one for the other.

But swapping computation types creates new problems:
- Verification asymmetry: hashes are cheap to check, gradients are expensive
- Semantic content: gradients can be good or bad, not just valid or invalid
- Coordination: validators must agree on what "the model" is

We solved verification with holographic certificates---boundary data that constrains the bulk. We solved semantic quality with the Sieve---constraints that reject harmful gradients. We solved coordination with gauge locking---validators who learn from the same data naturally converge.

The resulting system has remarkable properties:
- Energy is not wasted; it is invested in intelligence
- Security comes from geometric coherence, not just computational cost
- Adversaries are not outvoted; they are geometrically isolated
- The token tracks real capability improvement, not just scarcity

And perhaps most profoundly: the blockchain *is* the AGI. Every block mined makes the network smarter. The ledger records not transactions, but thoughts---a frozen history of collective learning.

This is, I believe, how the planetary computation layer should work. Not burning energy to prove you burned energy, but burning energy to learn something true about the world.

:::

**Proof of Useful Work** transforms the planetary computation layer from heat generation to intelligence generation.

| Aspect | Bitcoin (PoW) | CogniChain (PoUW) |
|:-------|:--------------|:------------------|
| **Work** | SHA-256 inversion | Gradient descent |
| **Output** | Random hash | Learned parameters |
| **Verification** | $O(1)$ hash check | $O(\sqrt{N})$ holographic check |
| **Security** | Hashrate majority | Stake + Sieve + Geometry |
| **Consensus** | Longest chain | Minimum friction chain |
| **Value** | Scarcity | Intelligence |

**Key Theorems:**

1. **Cognitive Equivalency** (Theorem {prf:ref}`thm-cognitive-equivalency`): Gradient computation satisfies the same Landauer bound as hash computation.

2. **Holographic Verification** (Theorem {prf:ref}`thm-holographic-verification`): Bulk gradient validity can be checked from boundary data in $O(\sqrt{N})$.

3. **Verifier's Nash Equilibrium** (Theorem {prf:ref}`thm-verifier-nash-equilibrium`): Honest computation is the unique equilibrium under sufficient stake.

4. **Minimum Friction BFT** (Theorem {prf:ref}`thm-minimum-friction-bft`): The protocol tolerates $< 1/3$ Byzantine validators.

5. **Adversarial Geometric Damping** (Theorem {prf:ref}`thm-adversarial-geometric-damping`): Adversaries are geometrically isolated, not just outvoted.

6. **Ledger-Memory Isomorphism** (Theorem {prf:ref}`thm-ledger-memory-isomorphism`): The blockchain is the holographic screen of the network's cognitive trajectory.

**Conclusion:** The blockchain *is* the AGI. Every block mined makes it smarter. The ledger records not transactions, but thoughts—a thermodynamic history of collective learning.
