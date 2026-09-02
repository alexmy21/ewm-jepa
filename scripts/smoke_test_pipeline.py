#!/usr/bin/env python
"""ewm-jepa pipeline smoke test.

Verifies the full loop on the RTX 3060:

    video -> V-JEPA encoder -> z
         -> quantize -> "tid{n}" stream
         -> HLLSet lattice (gate ∩, LUT, De Bruijn materialize)
         -> dequantize -> z_hat
         -> V-JEPA predictor -> prediction

Run:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
        /home/alexmy/.conda/envs/ewm-jepa/bin/python scripts/smoke_test_pipeline.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ewm_jepa import JEPAPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--codebook-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"[1/3] Building JEPAPipeline (codebook={args.codebook_size}) ...")
    pipe = JEPAPipeline(
        device=args.device,
        codebook_size=args.codebook_size,
        seed=args.seed,
    )
    print(f"      gate_TF HLLSet popcount: {pipe.gate.popcount()}")

    print("[2/3] Random 16-frame clip + vitl16.yaml masks ...")
    video, masks_enc, masks_pred = pipe.make_masks(batch_size=1)
    print(f"      video: {tuple(video.shape)}, mask pairs: {len(masks_enc)}")

    print("[3/3] Roundtrip: encoder -> lattice -> predictor ...")
    result = pipe.roundtrip(video, masks_enc, masks_pred)

    for i, stats in enumerate(result.stats):
        print(
            f"      pair {i}: ids {stats.input_ids} (unique {stats.unique_input_ids}) "
            f"-> gated set {stats.gated_ids} (set retention {stats.set_retention:.3f}), "
            f"ordered {stats.restored_ids} (retention {stats.retention:.3f})"
        )
        print(
            f"             HLLSet {stats.hllset_popcount} bits -> gate {stats.gate_popcount} bits | "
            f"LUT {stats.lut_size} | "
            f"quant cos {stats.quant_cosine:.3f} | pred cos {stats.pred_cosine:.3f}"
        )
        print(
            f"             pred shape original={tuple(result.pred_original[i].shape)} "
            f"restored={tuple(result.pred_restored[i].shape)}"
        )

    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
