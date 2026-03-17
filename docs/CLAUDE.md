# Claude Code Guidelines for Fragile Docs

## Project Overview

**Fragile** is a theoretical framework that derives cognitive architecture from first principles of bounded rationality. It connects information theory, control theory, differential geometry, and gauge theory to explain how agents with finite resources can act intelligently under uncertainty.

### Document Structure (10 Parts)

| Part | Folder | Theme |
|------|--------|-------|
| I | `01_foundations` | Core definitions: POMDP framework, bounded rationality, interaction under partial observability |
| II | `02_sieve` | Runtime safety: 60 diagnostic nodes, barrier methods, constraint satisfaction |
| III | `03_architecture` | Implementation: VQ-VAE with disentangled latents (macro/nuisance/texture) |
| IV | `04_control` | Belief dynamics: exploration, coupling windows, belief updates |
| V | `05_geometry` | Geometric dynamics: metric laws, WFR geometry, holographic generation, equations of motion |
| VI | `06_fields` | Field theory: boundary interfaces, reward fields, information bounds |
| VII | `07_cognition` | Cognitive extensions: supervised topology, governor, memory, ontology, metabolism, causality |
| VIII | `08_multiagent` | Gauge theory: Standard Model derivation, parameter sieve, constants verification |
| IX | `09_economics` | Economics: Proof of Useful Work (PoUW) consensus mechanism |
| X | `10_appendices` | Reference: derivations, parameter tables, FAQ, formal proofs |

### Core Concepts

- **Bounded Rationality Controller**: Agent with finite channel capacity $C$ between world and actions
- **The Sieve**: Runtime safety system with 60 diagnostic nodes ensuring constraint satisfaction
- **Latent Decomposition**: $Z_t = (K_t, Z_{n,t}, Z_{\mathrm{tex},t})$ — macro state, nuisance, texture
- **WFR Geometry**: Wasserstein-Fisher-Rao metric unifying transport and information geometry
- **Gauge Fields**: Opportunity ($B_\mu$), Error ($W_\mu$), Binding ($G_\mu$) fields from agent symmetries

---

## Agent Instructions

### Using the Feynman-Jupyter-Educator Agent

When creating or editing Feynman prose blocks, explanatory content, or adding intuitive explanations to documents, **always use the `feynman-jupyter-educator` agent**.

**Critical rules:**
- When editing **multiple documents**, launch **one agent per document in parallel** (single message with multiple Task tool calls)
- The agent understands the Feynman style and will properly format content with correct classes
- Never manually write Feynman blocks without using this agent

**Example - editing 3 documents:**
```
Launch 3 feynman-jupyter-educator agents in parallel:
- Agent 1: "Add Feynman explanation to section X in 01_definitions.md"
- Agent 2: "Add Feynman explanation to section Y in 02_diagnostics.md"
- Agent 3: "Add Feynman explanation to section Z in 03_compute_tiers.md"
```

---

## Feynman Agent Rules

When acting as the "Feynman agent" to add explanatory content, follow these strict rules.

### Content Classification

| Content Type | Class Required | Shown in Expert Mode? |
|--------------|----------------|----------------------|
| Formal definitions (`{prf:definition}`) | None | **YES** - always shown |
| Theorems (`{prf:theorem}`) | None | **YES** - always shown |
| Lemmas (`{prf:lemma}`) | None | **YES** - always shown |
| Corollaries (`{prf:corollary}`) | None | **YES** - always shown |
| Axioms (`{prf:axiom}`) | None | **YES** - always shown |
| Proofs (`{prf:proof}`) | None | **YES** - always shown |
| Feynman prose | `feynman-prose` | NO - hidden |
| Agent-added notes/examples | `feynman-added` | NO - hidden |
| Agent-added admonitions | `feynman-added` | NO - hidden |
| Agent-added tables (non-formal) | `feynman-added` | NO - hidden |

### What You CAN Do

1. **Write inside existing Feynman prose blocks:**
   ```markdown
   :::{div} feynman-prose
   [You can ADD content here]
   :::
   ```

2. **Create NEW Feynman prose blocks:**
   ```markdown
   :::{div} feynman-prose
   Your explanatory prose here. Write in Feynman's conversational style.
   :::
   ```

3. **Add NEW non-formal directives** (notes, examples, tips) - MUST use `feynman-added` class:
   ```markdown
   :::{note}
   :class: feynman-added
   This is an agent-added clarification that will be hidden in expert mode.
   :::
   ```

   ```markdown
   :::{admonition} Example: Computing the Metric
   :class: feynman-added tip
   Here's a worked example...
   :::
   ```

4. **Add NEW tables for intuition/examples** - MUST wrap in feynman-added div:
   ```markdown
   :::{div} feynman-added
   | Concept | Intuition | Formal Definition |
   |---------|-----------|-------------------|
   | ... | ... | ... |
   :::
   ```

5. **Add formal definitions that were MISSING** (no special class needed):
   ```markdown
   :::{prf:definition} Missing Concept Name
   :label: def-missing-concept

   Formal definition here...
   :::
   ```

### What You MUST NEVER Do

1. **NEVER delete existing content** - not formal definitions, not theorems, not prose, nothing
2. **NEVER modify text outside of Feynman prose blocks** - the formal mathematical content is immutable
3. **NEVER remove or alter existing Feynman prose** - only add to it or create new blocks
4. **NEVER add non-formal directives without the `feynman-added` class**
5. **NEVER change the structure of existing directives**

---

## Document Structure Patterns

### Heading Hierarchy

```
# Document Title (H1 - once per file)

(sec-section-name)=
## Major Section (H2 - 5-7 per document)

### Subsection (H3 - 2-4 per H2)

#### Deep Dive (H4 - sparingly)
```

**Rules:**
- Never use H5 or deeper - break into new H3 instead
- Always add section label anchor before H2: `(sec-descriptive-name)=`
- No formal definitions before H2 intro prose

### Section Structure Pattern

Each major section follows this order:

1. **Feynman prose introduction** (motivation, intuition)
2. **Formal definitions/theorems** (rigorous mathematics)
3. **Feynman prose interpretation** (what it means)
4. **Implementation notes** (Python code if applicable)
5. **Diagnostic connections** (references to Sieve nodes)
6. **Subsections** (H3)

### Researcher Bridge Boxes

Use for connecting to existing literature:

```markdown
(rb-concept-name)=
:::{admonition} Researcher Bridge: Topic Name
:class: info
Standard RL does X, but our framework does Y because...
:::
```

### Connection to RL Boxes

For mapping to standard RL concepts:

```markdown
:::{admonition} Connection to RL #N: Concept Name
:class: note
:name: conn-rl-N
This corresponds to [standard RL concept] because...
:::
```

---

## Mathematical Notation Conventions

### Core Symbols

| Category | Notation | Meaning |
|----------|----------|---------|
| **Latent state** | $Z_t = (K_t, Z_{n,t}, Z_{\mathrm{tex},t})$ | Macro, nuisance, texture decomposition |
| **Distributions** | $p(k\|x)$, $\bar{P}(k'\|k,a)$, $P_\partial$ | Posterior, transition kernel, boundary law |
| **Information** | $H(\cdot)$, $I(X;Y)$, $D_{\mathrm{KL}}(\cdot\|\cdot)$ | Entropy, mutual info, KL divergence |
| **Metrics** | $G_{ij}$, $d_G(\cdot,\cdot)$ | State-space metric, geodesic distance |
| **Gauge fields** | $B_\mu$, $W_\mu^a$, $G_\mu^a$ | Opportunity, Error, Binding fields |
| **Operators** | $\nabla$, $\operatorname{sg}[\cdot]$, $[\cdot]_+$ | Gradient, stop-gradient, ReLU |

### Subscript/Superscript Conventions

- Use `\mathrm{}` for semantic tags: $Z_{\mathrm{tex}}$, $D_{\mathrm{KL}}$
- Time indices: $t$ (interaction), $s$ (computation), $\tau$ (scale)
- Spatial indices: $i,j,k$ for latent, $\mu,\nu,\rho,\sigma$ for spacetime

### Unit Tracking

Always document units in mathematical exposition and code:
- $[\mathrm{nat}]$ - natural information units
- $[\mathrm{nat/step}]$ - information rate
- $[\mathrm{dimensionless}]$ - pure numbers

---

## Cross-Referencing System

### Label Patterns

| Type | Pattern | Example |
|------|---------|---------|
| Sections | `(sec-descriptive-name)=` | `(sec-bounded-rationality-controller)=` |
| Definitions | `def-concept-name` | `:label: def-belief-density` |
| Theorems | `thm-theorem-name` | `:label: thm-emergence-opportunity-field` |
| Corollaries | `cor-name` | `:label: cor-standard-model-symmetry` |
| Researcher Bridge | `(rb-concept)=` | `(rb-barriers-trust-regions)=` |
| RL Connections | `conn-rl-N` | `:name: conn-rl-22` |

### Reference Usage

```markdown
See {ref}`Section 2.8 <sec-conditional-independence>` for details.
This follows from {prf:ref}`def-bounded-rationality-controller`.
By {prf:ref}`thm-equivalence-entropy-regularized-control`, we have...
```

---

## Proof and Theorem Formatting

### Theorem Structure

```markdown
:::{prf:theorem} Theorem Name
:label: thm-short-name

[Statement of theorem]

*Proof.*

**Step 1. [Goal description]:**
$$[First equation]$$

**Step 2. [Next phase]:**
$$[Application or substitution]$$

**Step 3. [Computation]:**
$$[Explicit calculation]$$

**Identification:** [Physical/operational meaning of result]

**Remark:** [Extensions, caveats, special cases]

$\square$
:::
```

### Proof Patterns

- **By Steps:** 3-6 numbered phases, each with goal statement
- **By Variation:** Functional → first variation → Euler-Lagrange
- **By Bifurcation:** Expand potential → control parameter → stability analysis

---

## Feynman Prose Style Guide

### Opening Moves

- "Now we come to what I think is the most beautiful part..."
- "Here's the thing that trips people up..."
- "Ask yourself: why should..."
- "Let me tell you what this is really about..."
- "You might think X, but here's what's actually happening..."

### Rhetorical Devices

- **Thought experiments:** "Imagine you're a robot arm in a factory..."
- **Concrete failures:** "Suppose you train a vision-based robot and..."
- **Analogies:** "Like a well-designed software system with clear APIs..."
- **Counterintuition:** "This seems backwards, but..."
- **Progressive revelation:** "First... But here's the thing... And the key insight..."

### Closing Patterns

- "This explains why..."
- "And there it is."
- "This is what determines..."
- "Here is something that should make you sit up."

### Guidelines

- **Length:** 150-300 words per block
- **Placement:** Before formal definitions, after theorems
- **Density:** Interrupt dense math every 300-500 words
- **Tone:** Confident but not dogmatic, collaborative ("Let me make sure you understand...")

---

## Admonition Types

| Type | Class | Purpose |
|------|-------|---------|
| Researcher Bridge | `:class: info` | Connect to existing literature |
| Implementation tip | `:class: tip` | Practical coding guidance |
| Caveat/Warning | `:class: warning` | Critical limitations |
| Example | `:class: feynman-added example` | Worked examples (hidden in expert mode) |
| Note | `:class: feynman-added note` | Additional context (hidden in expert mode) |
| See Also | `:class: feynman-added seealso` | Cross-references (hidden in expert mode) |

---

## Code Block Conventions

### Python Style

```python
@dataclass
class DisentangledConfig:
    """Configuration for split-latent agent.

    Args:
        obs_dim: Observation dimension [pixels]
        macro_dim: Macro latent dimension $|K|$
        nuis_dim: Nuisance latent dimension $|Z_n|$
    """
    obs_dim: int = 64 * 64 * 3
    macro_dim: int = 512  # [nat]
    nuis_dim: int = 128   # [nat]
```

### Rules

- Type hints throughout
- Detailed docstrings with Args/Returns
- Variable names match math notation: `z_macro`, `K_t`, `z_nuis`
- Shape comments: `# [B, Z]` for tensor shapes
- Unit comments: `# [nat]`, `# [nat/step]`
- Always preceded by explanatory prose

---

## Verification Checklist

Before completing any edit:

- [ ] All new prose is inside `:::{div} feynman-prose` blocks
- [ ] All non-formal directives have `:class: feynman-added`
- [ ] Formal definitions/theorems have NO special class
- [ ] No existing content was deleted
- [ ] No formal definitions/theorems were modified
- [ ] Prose complements but doesn't duplicate formal content
- [ ] Labels follow naming conventions
- [ ] Cross-references use correct syntax
- [ ] Mathematical notation matches established conventions

---

## Quick Reference

```markdown
# Prose block (hidden in expert mode)
:::{div} feynman-prose
Conversational explanation...
:::

# Agent-added note (hidden in expert mode)
:::{note}
:class: feynman-added
Clarification...
:::

# Agent-added example (hidden in expert mode)
:::{admonition} Example Title
:class: feynman-added example
Worked example...
:::

# Researcher Bridge (always shown)
:::{admonition} Researcher Bridge: Topic
:class: info
Literature connection...
:::

# Formal definition (ALWAYS shown - no class)
:::{prf:definition} Name
:label: def-name
Formal content...
:::

# Theorem with proof (ALWAYS shown)
:::{prf:theorem} Name
:label: thm-name
Statement...

*Proof.*
**Step 1.** ...
$\square$
:::

# Section with label
(sec-section-name)=
## Section Title

# Cross-references
{ref}`Section Title <sec-section-name>`
{prf:ref}`def-name`
{prf:ref}`thm-name`
```
