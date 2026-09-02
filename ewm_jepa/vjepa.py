# ewm_jepa/vjepa.py
"""V-JEPA model loading and forward helpers for ewm-jepa.

Loads the official facebookresearch/jepa ViT-L/16 encoder + predictor and
exposes the exact hooks the EWM lattice sits between (see README data
boundary):

    video -> encoder -> z (context encodings)   \
    video -> encoder -> h (target encodings)     -> predictor -> prediction
                      z passes through the HLLSet lattice in JEPAPipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JEPA_ROOT = PROJECT_ROOT / "jepa"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "vitl16.pth.tar"

if str(JEPA_ROOT) not in sys.path:
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


def _strip_module_prefix(state_dict: dict) -> dict:
    """Official checkpoints were saved from DDP-wrapped models."""
    if any(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_vjepa(
    device: str = "cuda:0",
    checkpoint: str | Path | None = None,
    num_frames: int = 16,
    crop_size: int = 224,
):
    """Build V-JEPA ViT-L/16 encoder + predictor and load official weights.

    Returns (encoder, predictor) in eval mode on ``device``.
    """
    checkpoint = Path(checkpoint) if checkpoint is not None else CHECKPOINT

    encoder, predictor = init_video_model(
        device=torch.device(device),
        patch_size=16,
        num_frames=num_frames,
        tubelet_size=2,
        model_name="vit_large",
        crop_size=crop_size,
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

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing_e, unexpected_e = encoder.load_state_dict(
        _strip_module_prefix(ckpt["encoder"]))
    missing_p, unexpected_p = predictor.load_state_dict(
        _strip_module_prefix(ckpt["predictor"]))
    del ckpt

    if missing_e or missing_p or unexpected_e or unexpected_p:
        raise RuntimeError(
            f"checkpoint mismatch: enc missing={len(missing_e)} "
            f"unexpected={len(unexpected_e)}; pred missing={len(missing_p)} "
            f"unexpected={len(unexpected_p)}")

    return encoder, predictor


def make_mask_collator(num_frames: int = 16, crop_size: int = 224) -> MaskCollator:
    """Mask collator with the exact vitl16.yaml mask configs."""
    return MaskCollator(
        cfgs_mask=MASK_CONFIGS,
        crop_size=crop_size,
        num_frames=num_frames,
        patch_size=16,
        tubelet_size=2,
    )


def random_video_and_masks(
    batch_size: int = 1,
    num_frames: int = 16,
    crop_size: int = 224,
    device: str = "cuda:0",
    seed: int = 0,
):
    """Random 16-frame clip plus encoder/predictor masks (as in training)."""
    torch.manual_seed(seed)
    collator = make_mask_collator(num_frames=num_frames, crop_size=crop_size)
    samples = [
        torch.randn(3, num_frames, crop_size, crop_size)
        for _ in range(batch_size)
    ]
    video, masks_enc, masks_pred = collator(samples)
    video = video.to(device)
    masks_enc = [m.to(device) for m in masks_enc]
    masks_pred = [m.to(device) for m in masks_pred]
    return video, masks_enc, masks_pred


def encode_full(encoder, video):
    """Full spatio-temporal patch encodings, shape (B, N, D)."""
    with torch.no_grad():
        return encoder(video)


def encode_masked(encoder, video, masks_enc):
    """Context encodings per mask (list of (B, K, D))."""
    with torch.no_grad():
        return encoder(video, masks_enc)


def target_encodings(encoder, video, masks_pred):
    """Target encodings per mask (list of (B, K_tgt, D))."""
    with torch.no_grad():
        h_full = encoder(video)
        return apply_masks(h_full, masks_pred, concat=False)


def predict(predictor, z_list, h_list, masks_enc, masks_pred):
    """Latent prediction per mask (list of (B, K_tgt, D))."""
    with torch.no_grad():
        return predictor(z_list, h_list, masks_enc, masks_pred)
