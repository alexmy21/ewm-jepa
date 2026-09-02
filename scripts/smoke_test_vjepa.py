#!/usr/bin/env python
"""V-JEPA ViT-L/16 GPU smoke test for ewm-jepa.

Loads the pretrained V-JEPA encoder + predictor and runs a random
16-frame clip through the full latent-prediction pipeline to verify
the environment works on the RTX 3060.

Run:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
        /home/alexmy/.conda/envs/ewm-jepa/bin/python scripts/smoke_test_vjepa.py

(PCI_BUS_ID + CUDA_VISIBLE_DEVICES=1 selects the RTX 3060 shown as
GPU 1 in nvidia-smi. The Quadro M1200 is GPU 0.)
"""

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JEPA_ROOT = PROJECT_ROOT / "jepa"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "vitl16.pth.tar"

sys.path.insert(0, str(JEPA_ROOT))

from app.vjepa.utils import init_video_model  # noqa: E402
from src.masks.multiblock3d import MaskCollator  # noqa: E402
from src.masks.utils import apply_masks  # noqa: E402


# Mask configs from jepa/configs/pretrain/vitl16.yaml
MASK_CONFIGS = [
    {
        "aspect_ratio": (0.75, 1.5),
        "num_blocks": 8,
        "spatial_scale": (0.15, 0.15),
        "temporal_scale": (1.0, 1.0),
        "max_temporal_keep": 1.0,
        "max_keep": None,
    },
    {
        "aspect_ratio": (0.75, 1.5),
        "num_blocks": 2,
        "spatial_scale": (0.7, 0.7),
        "temporal_scale": (1.0, 1.0),
        "max_temporal_keep": 1.0,
        "max_keep": None,
    },
]


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def strip_module_prefix(state_dict: dict) -> dict:
    """Checkpoints were saved from DDP-wrapped models ('module.' keys)."""
    if any(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting.")
        return 1

    print(f"[1/6] Building V-JEPA ViT-L/16 (num_frames={args.num_frames}, "
          f"crop_size={args.crop_size}) ...")
    encoder, predictor = init_video_model(
        device=device,
        patch_size=16,
        num_frames=args.num_frames,
        tubelet_size=2,
        model_name="vit_large",
        crop_size=args.crop_size,
        pred_depth=12,
        pred_embed_dim=384,
        uniform_power=True,
        use_mask_tokens=True,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_sdpa=True,
    )
    encoder.eval()
    predictor.eval()
    print(f"    Encoder params:  {count_parameters(encoder)/1e6:.1f} M")
    print(f"    Predictor params: {count_parameters(predictor)/1e6:.1f} M")

    print(f"[2/6] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing_e, unexpected_e = encoder.load_state_dict(
        strip_module_prefix(ckpt["encoder"]))
    missing_p, unexpected_p = predictor.load_state_dict(
        strip_module_prefix(ckpt["predictor"]))
    print(f"    encoder  missing={len(missing_e)} unexpected={len(unexpected_e)}")
    print(f"    predictor missing={len(missing_p)} unexpected={len(unexpected_p)}")
    del ckpt  # free ~5 GB of CPU RAM

    print("[3/6] Building masks (as in vitl16.yaml) ...")
    collator = MaskCollator(
        cfgs_mask=MASK_CONFIGS,
        crop_size=args.crop_size,
        num_frames=args.num_frames,
        patch_size=16,
        tubelet_size=2,
    )
    sample = torch.randn(3, args.num_frames, args.crop_size, args.crop_size)
    video, masks_enc, masks_pred = collator([sample])
    video = video.to(device)
    masks_enc = [m.to(device) for m in masks_enc]
    masks_pred = [m.to(device) for m in masks_pred]
    print(f"    video shape: {tuple(video.shape)}")
    for i, (me, mp) in enumerate(zip(masks_enc, masks_pred)):
        print(f"    mask pair {i}: encoder keep={me.numel()}, predictor target={mp.numel()}")

    print("[4/6] Forward pass (encoder -> target encodings -> predictor) ...")
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        t0 = time.time()
        z = encoder(video, masks_enc)                      # context encodings (list)
        h_full = encoder(video)                            # full target encodings
        h = apply_masks(h_full, masks_pred, concat=False)  # target encodings (list)
        pred = predictor(z, h, masks_enc, masks_pred)      # latent predictions (list)
        torch.cuda.synchronize(device)
        elapsed = time.time() - t0

    print(f"    elapsed: {elapsed*1000:.0f} ms")
    for i, (zi, hi, pi) in enumerate(zip(z, h, pred)):
        print(f"    pair {i}: context {tuple(zi.shape)} -> target {tuple(hi.shape)} "
              f"-> prediction {tuple(pi.shape)}")

    print("[5/6] GPU summary")
    props = torch.cuda.get_device_properties(device)
    print(f"    device: {torch.cuda.get_device_name(device)}")
    print(f"    VRAM:   {props.total_memory/1e9:.1f} GB")
    print(f"    peak allocated: {torch.cuda.max_memory_allocated(device)/1e9:.2f} GB")
    print(f"    peak reserved:  {torch.cuda.max_memory_reserved(device)/1e9:.2f} GB")

    print("[6/6] Smoke test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
