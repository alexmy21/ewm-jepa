# ewm-jepa

## A message before the installation guide

> *Professor LeCun — your JEPA program asked the right question: a world model
> should learn by predicting in latent space, not by reconstructing pixels.
> We believe we have built a candidate answer to the question your program
> leaves open: what is the **memory** of a JEPA made of?*

**V-JEPA perceives.** It turns video into 1568 spatio-temporal patch encodings
and predicts missing regions in latent space. Its ontology, however, is fixed
by design — the patch grid, the embedding dimensions — and its memory is
parametric: weights that are overwritten by every gradient step, which is why
neural networks forget catastrophically.

**Emerging World Models (EWM) remember.** EWM — consolidated in
[`docs/EWM/main.tex`](docs/EWM/main.tex) — replaces the predefined vocabulary
with an **emergent ontology**: elements are not assigned by a designer, they
are *recognized* when measurement patterns persist. Its memory is an **HLLSet
lattice** — 32,768-bit holographic plates that are Idempotent, Immutable, and
Content-Addressed (IICA). Its learning is **rank rearrangement**, not weight
updates. Its evolution obeys **Noether conservation laws** (the DRN balance:
Departed / Retained / Novel), and its intrinsic reward is **uncertainty
reduction** — the same opponent-process signal biology uses, per the Lincoln
blueprint in the appendix of `main.tex`.

**ewm-jepa is the handshake.** We place the EWM lattice *between* the V-JEPA
encoder and predictor:

```text
video ──► V-JEPA encoder ──► z (1568 × 1024, continuous)
                                │
                                ▼  quantize (deterministic codebook)
                         encoding IDs  "tid42 tid671 ..."
                                │
                                ▼  HLLSet lattice (IICA)
                     ingest ∪/∩ · gate_TF · LUT · DRN · BSS · temporal pyramid
                                │
                                ▼  materialize (TF-ranked, De Bruijn)
                         restored  "tid42 tid671 ..."
                                │
                                ▼  dequantize (codebook centroids)
                         ẑ (restored encodings)
                                │
                                ▼
                V-JEPA predictor ──► final latent prediction
```

This is exactly the architecture of our DeepSeek-OCR extension
(`hllset_cortex`), with the OCR front end replaced by V-JEPA. The EWM
boundary is **encoding-agnostic**: whether `tid671` is a BPE token or a
quantized patch encoding, MurmurHash3 treats it as an opaque measurement.

**The complementarity, in one sentence:** JEPA supplies the perceptual
substrate and the forward model; EWM supplies the emergent ontology, the
holographic memory, and the executive. Neither is complete alone.

The full mapping — four EWM principles and the three-tier perceptron taxonomy
instantiated in this repository — is in
[`docs/NARRATIVE.md`](docs/NARRATIVE.md). The notebooks in
[`notebooks/`](notebooks/) are the working evidence.

---

## Status

- V-JEPA **ViT-L/16** runs on the RTX 3060 (smoke test passes: 545 ms forward,
  1.41 GB peak VRAM).
- The **whole hllset-cortex is ported into this repo** (`hllset_cortex/`:
  Python package + vendored Rust crate + docs + original notebooks).  The
  env builds `hllset-py` from the vendored crate and installs the local
  `hllset_cortex` editable — no external paths.
- The HLLSet-Cortex stack is wired to V-JEPA through a deterministic
  quantizer (`ewm_jepa/`).
- Six notebooks reproduce the DeepSeek-OCR/hllset_cortex experiments with a
  V-JEPA front end.

## Hardware

| GPU (nvidia-smi index) | Name | VRAM | Role |
| ----------------------- | ---------------- | ----- | --------------- |
| 0 | Quadro M1200 | 4 GB | display |
| 1 | GeForce RTX 3060 | 12 GB | V-JEPA (use this) |

Always select the RTX 3060 explicitly:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
```

## Notebooks — the complementary ways, demonstrated

Each notebook mirrors its DeepSeek-OCR/hllset_cortex counterpart, with the
OCR encoder replaced by the V-JEPA encoder and the OCR decoder replaced by the
V-JEPA predictor.

| # | Notebook | Mirrors | What it shows |
| --- | --- | --- | --- |
| 1 | `01_jepa_hllset_pipeline.ipynb` | `01_ocr_hllset_pipeline_real.ipynb` | Full round-trip: encoder → quantize → HLLSet → LUT → materialize → predictor |
| 2 | `02_jepa_ewm_unification.ipynb` | `02_caal_cortex_unification.ipynb` | Three-LUT architecture; cross-LUT disambiguation; De Bruijn order restoration |
| 3 | `03_recursive_iica_chain.ipynb` | `03_recursive_iica_chain.ipynb` | The chained [UM]: IICA composition, progressive gates, feedback convergence |
| 4 | `04_um_net_agent_network.ipynb` | `04_dg_agent_network.ipynb` | [UM]-Net: fire-and-forget agent graphs over encoding streams |
| 5 | `05_holographic_memory.ipynb` | `08_holographic_memory.ipynb` | Temporal pyramid; holographic reconstruction; TF time lens |
| 6 | `06_grounding_proof.ipynb` | `prove_ewm_grounding.ipynb` | Fidelity round-trip and one-sided grounding with real V-JEPA encodings |

Run them:

```bash
cd /home/alexmy/SGS/SGS_lib/fractal_manifold/ewm-jepa
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    /home/alexmy/.conda/envs/ewm-jepa/bin/jupyter notebook notebooks/
```

## Environment setup

```bash
# 1. conda env
conda create -n ewm-jepa python=3.10 -y -c conda-forge

# 2. PyTorch (cu124, proven with driver 565.77 / CUDA 12.7)
/home/alexmy/.conda/envs/ewm-jepa/bin/pip install \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. V-JEPA dependencies
cd /home/alexmy/SGS/SGS_lib/fractal_manifold/ewm-jepa
/home/alexmy/.conda/envs/ewm-jepa/bin/pip install -r jepa/requirements.txt

# 4. HLLSet stack — fully ported into this repo (see hllset_cortex/)
#    Build the Rust wheel from the vendored crate, then install the local
#    Python package editable.  (maturin: `pip install maturin` once.)
cd /home/alexmy/SGS/SGS_lib/fractal_manifold/ewm-jepa
maturin build --release -q --manifest-path \
    hllset_cortex/crates/hllset_py/Cargo.toml
/home/alexmy/.conda/envs/ewm-jepa/bin/pip install --force-reinstall \
    hllset_cortex/crates/hllset_py/target/wheels/hllset_py-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
/home/alexmy/.conda/envs/ewm-jepa/bin/pip install -e hllset_cortex

# 5. Pretrained V-JEPA ViT-L/16 checkpoint (5.1 GB; we use encoder + predictor)
mkdir -p checkpoints
curl -L -o checkpoints/vitl16.pth.tar \
    https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar
```

## Smoke test

```bash
cd /home/alexmy/SGS/SGS_lib/fractal_manifold/ewm-jepa
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    /home/alexmy/.conda/envs/ewm-jepa/bin/python scripts/smoke_test_vjepa.py
```

Expected: ViT-L/16 encoder (305.5 M) + predictor (22.7 M) load the official
checkpoint with 0 missing/0 unexpected weights; a random 16-frame clip is
encoded and predicted in ~0.5 s; peak VRAM is ~1.4 GB.

## Repository layout

```text
ewm-jepa/
├── docs/
│   ├── EWM/main.tex       # canonical EWM consolidation document (the narrative)
│   └── NARRATIVE.md       # EWM principles ↔ V-JEPA components mapping
├── notebooks/             # the six reproduced notebooks (see table above)
├── ewm_jepa/              # shared package: vjepa loader, quantizer, JEPAPipeline
├── hllset_cortex/         # full hllset-cortex port (Python pkg + Rust crate +
│                          #   docs + original notebooks), self-contained
├── scripts/
│   ├── smoke_test_vjepa.py
│   ├── smoke_test_pipeline.py
│   └── build_notebooks.py # regenerates the notebooks
├── jepa/                  # upstream facebookresearch/jepa clone (gitignored)
├── checkpoints/           # pretrained weights (gitignored)
└── README.md              # this file — the message, then the manual
```

## References

- **EWM consolidation:** [`docs/EWM/main.tex`](docs/EWM/main.tex) (four
  principles, perceptron taxonomy, ds-hllset-cortex appendix)
- **HLLSet theory:** A. Mylnikov, *HLLSet Theory: A Unified Framework for
  Probabilistic Knowledge Representation*, ASTESJ 11(2), 2026
  ([pdf](https://www.astesj.com/publications/ASTESJ_110202.pdf))
- **JEPA:** A. Bardes et al., *Revisiting Feature Prediction for Learning
  Visual Representations from Video* (V-JEPA),
  [facebookresearch/jepa](https://github.com/facebookresearch/jepa)
- **World-model program:** Y. LeCun, *A Path Towards Autonomous Machine
  Intelligence*, Open Review, 2022
  ([openreview.net](https://openreview.net/forum?id=BZ5a1r-kVsf))
- **Uncertainty reduction as primary reinforcer:** A. Lincoln, *Academia
  Neuroscience and Brain Research*, Vol. 2, 2026
  ([doi](https://doi.org/10.20935/AcadNeurosci8451))
