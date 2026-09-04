# EWM × V-JEPA — The Complementarity Narrative

**Canonical source:** [`docs/EWM/main.tex`](EWM/main.tex) — *Emerging World Models:
A Structural Framework for Self-Organizing World Representations* (Aug 2026).
Every document, notebook, and diagram in this repository is a footnote to that
document. This file is the translation layer: it maps the EWM principles onto
V-JEPA components.

---

## The thesis, in one paragraph

LeCun's JEPA gives us the right *perceptual question*: learn by predicting in
latent space, not by reconstructing pixels. EWM gives us the right *memory
answer*: a lattice of immutable, content-addressed measurements whose ontology
emerges from observation and whose evolution obeys conservation laws instead
of gradient descent. **V-JEPA supplies the substrate and the forward model;
EWM supplies the emergent ontology, the holographic memory, and the executive.**
Neither is complete alone — together they form a world model that perceives
like JEPA and remembers like a lattice.

## What each side contributes

| Concern | V-JEPA (LeCun) | EWM (this work) | ewm-jepa |
| --- | --- | --- | --- |
| Perception | Pretrained ViT encoder, spatio-temporal patches | Content-blind hashing of whatever enters the lattice | encoder → quantize → `tid{n}` |
| Prediction | Latent predictor (the "decoder") | DRN decomposition; BSS opponent-process signals | lattice → materialize → predictor |
| Memory | Parametric (weights) | HLLSet lattice + LUT, content-addressed (SHA-1 CIDs) | lattice between encoder and predictor |
| Ontology | Predefined (patch grid, embed dims) | **Emergent** (what persists across measurement) | LUT over quantized encodings |
| Learning | Gradient descent over weights | **Rank rearrangement** in the LUT; thresholds | both, at different layers |
| Forgetting | Catastrophic (weights overwritten) | **Exclusion without erasure** | the lattice never forgets |
| Reward signal | Contrastive/energy objectives | **Uncertainty reduction** (DRN/BSS) | intrinsic signal from the lattice |
| Scale law | Parameters + data | **IICA composition** (nodes compose without coordination) | [UM]-Net over encodings |

## The four EWM principles, mapped to V-JEPA

### Principle 1 — Emergent Ontology

> The elements of a world model are not defined a priori. They emerge from
> the accumulation of measurement interactions.

V-JEPA's ontology is fixed by design: a 16×224×224 clip becomes 1568 patch
tokens of 1024 dims. EWM's ontology is fixed by *nothing*: the LUT accumulates
every encoding ID that ever appears, and an element is simply *what persists*.
In ewm-jepa the quantizer is the bridge — it projects continuous patch
encodings onto a codebook of `tid{n}` IDs, and from that moment on the EWM
machinery treats them exactly as it treated DeepSeek-OCR's `tid671`. New
visual concepts need no vocabulary change; they are hashed, their bits are
set, and they accumulate TF until they emerge.

### Principle 2 — IICA Morphisms

> Every relation is Idempotent, Immutable, and Content-Addressed; composition
> preserves all three.

V-JEPA's forward pass is deterministic but parametric — change a weight and
the morphism changes. EWM's pipeline (hash → HLLSet → gate ∩ → LUT →
materialize) is IICA. In ewm-jepa the *whole loop* becomes IICA **between
commits**: with the lattice frozen, the same video always produces the same
`tid` stream, the same HLLSet, the same restored encodings, the same
prediction. The lattice state itself changes only at explicit commit events.
This is what makes the system auditable: every world state is a SHA-1
address.

### Principle 3 — Ashby–Bootstrap Correspondence

> Mismatched representational capacity is bridged by multiplying
> measurements, not by reducing resolution.

V-JEPA's encoder output is high-dimensional and continuous (a capacity
mismatch in Ashby terms: the continuous encoder vs. the finite bit lattice).
The quantizer compresses; the n-gram tokenizer (1-, 2-, 3-grams over the
`tid` stream) then *multiplies* measurements — the same patch sequence is
viewed as unigrams, bigrams, and trigrams, exactly Khayyam's trick of many
crude observations beating one fine instrument. The union of these views is
the HLLSet fingerprint.

### Principle 4 — Noether Evolution

> Structural change obeys conservation laws; every symmetry of the
> measurement regime yields a conserved quantity.

V-JEPA learns by weight updates — destructive, parametric. EWM evolves by
DRN decomposition (Departed / Retained / Novel) under the steering condition
`|card(N) − card(D)| → 0`, and by the temporal pyramid's union invariant:
information is never lost from the system state, only redistributed across
layers. In ewm-jepa, gradient descent remains where it belongs (the frozen
pretrained encoder/predictor), while the memory layer evolves under Noether
constraints — learning as *attention redistribution*, not erasure.

## The perceptron taxonomy, instantiated in ewm-jepa

From the `main.tex` appendix (Lincoln's uncertainty-reduction blueprint):

| Tier | EWM definition | ds-hllset-cortex component | **ewm-jepa component** |
| --- | --- | --- | --- |
| Type 0 — Substrate | Hash + bitmask accumulation, content-blind | ds-Encoder + MurmurHash3 | **V-JEPA encoder + quantizer** |
| Type 1 — Forward | Bootstrap views, temporal pyramid | Tokenizer + gate + LUT | **n-gram tokenizer + gate ∩ + LUT** |
| Type 2 — Executive | Rank assignment, Noether steering | Materializer + ds-Decoder | **materializer + V-JEPA predictor** |

The prediction-error signals carry over unchanged: a bit transition `0→1` is
novelty, `1→1` is confirmation; directed BSS gives the opponent-process triad
(τ = confirmation, ρ = departure, reverse-ρ = novelty); the matching law
emerges from five-level rank propagation. The only thing that changed between
DeepSeek-OCR and V-JEPA is the *front end* — which is exactly the point: the
EWM boundary is encoding-agnostic.

## The data boundary

```text
video ──► V-JEPA encoder ──► z (1568 × 1024, continuous)
                                │
                                ▼  quantize (codebook, deterministic)
                         encoding IDs "tid42 tid671 ..."
                                │
                                ▼  HLLSet lattice (IICA)
                     ingest ∪/∩, gate_TF, LUT, DRN, BSS, pyramid
                                │
                                ▼  materialize (n-gram disambiguation, TF tie-break)
                         restored "tid42 tid671 ..."
                                │
                                ▼  dequantize (codebook centroids)
                         ẑ (1568 × 1024, restored)
                                │
                                ▼
                V-JEPA predictor ──► final latent prediction
```

Same shape as `hllset_cortex` for DeepSeek-OCR; only stages 1 and 8 are
model-specific.

## What each side gains

**JEPA gains from EWM**

1. **A memory that does not forget.** HLLSets are immutable; excluded states
   are re-admissible at their earned TF — no catastrophic forgetting.
2. **A gradient-free learning layer.** Rank rearrangement + threshold tuning
   are cheap, auditable, and FPGA-friendly.
3. **An intrinsic reinforcement signal.** DRN/BSS uncertainty reduction, with
   the Lincoln correspondence as biological validation.
4. **Composition without coordination.** [UM]-Net nodes over encoding streams
   converge by CRDT union — no consensus, no backprop.
5. **An emergent vocabulary.** The system can express what it has *measured*,
   not merely what its pretrained codebook contained.

**EWM gains from JEPA**

1. **A real perceptual substrate.** The V-JEPA encoder is a world-class
   spatio-temporal feature extractor; EWM no longer needs text tokens.
2. **Latent-space prediction.** The V-JEPA predictor is the forward model at
   the top of the loop — the "decoder" that turns restored encodings into a
   testable prediction.
3. **A path to LeCun's program.** JEPA's stated goal — world models that plan
   and reason — needs a memory; EWM is a candidate answer for what that
   memory is made of.

## Open questions this project is designed to answer

1. What is the optimal quantizer (codebook size, bootstrap order) for
   V-JEPA patch encodings?
2. Does the gate-as-world-view dynamic (latent vocabulary, belief revision)
   hold when the "vocabulary" is a learned visual codebook?
3. Can the predictor's latent predictions be steered by the Noether
   controller across temporal layers?
4. Does uncertainty reduction (DRN/BSS) drive useful exploration in a
   JEPA-perceived environment?
