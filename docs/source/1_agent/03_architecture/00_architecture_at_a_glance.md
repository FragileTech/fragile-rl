(sec-architecture-at-a-glance)=
# Architecture at a Glance

## TLDR

- The representation stack is the TopoEncoder (Attentive Atlas) with typed latents: chart id $K$,
  nuisance $z_n$, texture $z_{\mathrm{tex}}$, and geometry $z_{\mathrm{geo}}$.
- Routing is chart-based (CovariantChartRouter or dot-product fallback) and produces chart weights
  that gate codebooks and decoder projectors.
- Each chart has its own codebook; optional SoftEquivariant metrics and soft straight-through
  assignments shape distances and gradients.
- Decoding uses chart projectors + a shared renderer, with a separate texture residual path.
- Training combines reconstruction + VQ + routing/consistency terms, with optional jump and
  supervised topology losses.

## Pipeline Overview

```{mermaid}
%%{init: {"themeVariables": {"background":"#0b111b","edgeLabelBackground":"#111827","textColor":"#e5e7eb","lineColor":"#9ca3af","primaryColor":"#1f2937","primaryTextColor":"#e5e7eb","clusterBkg":"#0f172a","clusterBorder":"#334155"}}}%%
flowchart TD
    subgraph ENC["Encoder + Atlas (TopoEncoderPrimitives)"]
        X["input x"] --> FE["Feature extractor\nMLP or CovariantRetina"]
        FE --> V["val_proj -> v"]
        ChartCenters["chart_centers"] --> Router["Chart router\nCovariantChartRouter or dot-product"]
        FE --> Router
        V --> Router
        Router --> Wenc["w_enc"]
        Router --> Kchart["K_chart"]
        Wenc --> Cbar["c_bar"]
        V --> Vlocal["v_local = v - c_bar"]
        Cbar --> Vlocal
        Codebook["codebook per chart"] --> VQ["per-chart VQ\n(+ soft equiv metric)"]
        Vlocal --> VQ
        VQ --> Zgeo["z_geo"]
        VQ --> Ztex["z_tex"]
        VQ --> Zn["z_n"]
    end

    subgraph DEC["Decoder (PrimitiveTopologicalDecoder)"]
        Zgeo --> DecRouter["Chart router\nCovariantChartRouter or latent_router"]
        DecRouter --> Mix["chart_projectors + gate\nweighted mix"]
        Mix --> Render["renderer + skip"]
        Ztex --> Tex["tex_residual"]
        Render --> Add["x_hat"]
        Tex --> Add
    end

    Zn --> Jump["Jump operator (optional)"]
    Wenc --> Sup["Supervised topology (optional)"]
    Zgeo --> Sup
    Wenc --> Cls["Invariant classifier (optional)"]
    Zgeo --> Cls
```

## Module Map

```{mermaid}
%%{init: {"themeVariables": {"background":"#0b111b","edgeLabelBackground":"#111827","textColor":"#e5e7eb","lineColor":"#9ca3af","primaryColor":"#1f2937","primaryTextColor":"#e5e7eb","clusterBkg":"#0f172a","clusterBorder":"#334155"}}}%%
flowchart LR
    subgraph TOP["TopoEncoderPrimitives"]
        Enc["PrimitiveAttentiveAtlasEncoder"]
        Dec["PrimitiveTopologicalDecoder"]
    end

    Enc --> Router["CovariantChartRouter"]
    Dec --> Router
    Enc --> SoftEq["SoftEquivariantLayer (optional)"]
    Enc --> Retina["CovariantRetina (optional)"]

    TOP --> Jump["FactorizedJumpOperator (optional)"]
    TOP --> SupLoss["SupervisedTopologyLoss (optional)"]
    TOP --> Cls["InvariantChartClassifier (optional)"]
```

## Where to Go Next

- Detailed block diagrams: {ref}`TopoEncoder architecture <sec-topoencoder-architecture>`
- World model attention: {ref}`Covariant cross-attention <sec-covariant-cross-attention-architecture>`
- Compute tiers and deployment tradeoffs: {ref}`Compute tiers <sec-computational-considerations>`
