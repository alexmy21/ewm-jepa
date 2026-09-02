# ewm-jepa

Apply the HLLSet-Cortex idea (from `DeepSeek-OCR-hllset/hllset_cortex`) to
[facebookresearch/jepa](https://github.com/facebookresearch/jepa) (V-JEPA):

- **Intercept** V-JEPA encoder outputs (spatio-temporal patch encodings)
- **Build an HLLSet Lattice** over (discretized) encoding IDs
- **Logical processing** on the lattice (union / intersection / BSS / LUT gates)
- **Materialize** the resulting HLLSet back into V-JEPA encoding space
- Feed the **V-JEPA predictor** ("decoder") to produce the final latent result

## Status

Environment setup + GPU smoke test for **V-JEPA ViT-L/16** on the RTX 3060.

## Hardware

| GPU (nvidia-smi index) | Name | VRAM | Role |
| ----------------------- | ----------------- | ------- | ---------------- |
| 0 | Quadro M1200 | 4 GB | display |
| 1 | GeForce RTX 3060 | 12 GB | JEPA model (use this) |

Always select the RTX 3060 explicitly:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
```

## Environment setup

```bash
# 1. conda env
conda create -n ewm-jepa python=3.10 -y -c conda-forge

# 2. PyTorch (cu124 is proven on this machine with driver 565.77 / CUDA 12.7)
/home/alexmy/.conda/envs/ewm-jepa/bin/pip install \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. JEPA dependencies
cd /home/alexmy/SGS/SGS_lib/fractal_manifold/ewm-jepa
/home/alexmy/.conda/envs/ewm-jepa/bin/pip install -r jepa/requirements.txt

# 4. Pretrained V-JEPA ViT-L/16 checkpoint (5.1 GB, contains encoder +
#    target-encoder EMA + optimizer state; we only need encoder/predictor)
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

Expected: encoder + predictor build with ViT-L/16 (1024-dim, 24 layers),
a random 16-frame 224x224 clip is encoded, context/target masks are applied,
the predictor predicts target encodings, and peak VRAM stays well below 12 GB.

## Layout

```text
ewm-jepa/
├── jepa/                  # upstream facebookresearch/jepa clone (gitignored)
├── checkpoints/           # pretrained weights (gitignored)
├── scripts/
│   └── smoke_test_vjepa.py
└── README.md
```

## Interception points (for the next step)

From `app/vjepa/train.py` (the hook location is the same in inference):

```python
z = encoder(video, masks_enc)             # context encodings   [B, K, 1024]
h = apply_masks(encoder(video), masks_pred, concat=False)  # target encodings
pred = predictor(z, h, masks_enc, masks_pred)              # latent prediction
```

The HLLSet-Cortex layer will sit between `encoder` and `predictor`:

```text
video ──> encoder ──> z (1024-dim patch encodings)
                        │
                        ▼
              discretize/quantize ──> encoding IDs (tid{n})
                        │
                        ▼
              HLLSet Lattice (ingest ∪/∩, LUT, BSS τ/ρ)
                        │
                        ▼
              materialize ──> restored 1024-dim encodings
                        │
                        ▼
              predictor ──> final latent result
```
