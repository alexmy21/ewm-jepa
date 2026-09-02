# ewm_jepa/quantize.py
"""Deterministic quantizer bridging continuous V-JEPA encodings and EWM IDs.

The EWM boundary is encoding-agnostic: HLLSet Algebra only needs opaque
``tid{n}`` strings.  V-JEPA, however, emits continuous 1024-dim patch
encodings.  This module is the bridge:

    z (K, D)  --encode-->  ids (K,)  --ids_to_text-->  "tid42 tid671 ..."
    "tid42 tid671 ..." --tid_to_ids--> ids --decode-->  z_hat (K, D)

The codebook is a fixed set of unit-norm anchors (a frozen, content-addressed
vocabulary).  Encoding is nearest-anchor in cosine space; decoding returns
the anchor scaled by the source vector's norm, so both direction and
magnitude are preserved as faithfully as the codebook allows.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class Quantizer:
    """Cosine-nearest-anchor quantizer over a frozen codebook."""

    def __init__(
        self,
        dim: int = 1024,
        codebook_size: int = 4096,
        seed: int = 0,
        device: str = "cuda:0",
    ):
        self.dim = dim
        self.codebook_size = codebook_size
        self.device = torch.device(device)

        generator = torch.Generator().manual_seed(seed)
        anchors = torch.randn(codebook_size, dim, generator=generator)
        self.anchors = F.normalize(anchors, dim=-1).to(self.device)

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., D) -> ids: (...) in [0, codebook_size)."""
        zn = F.normalize(z.float(), dim=-1)
        sim = zn @ self.anchors.T
        return sim.argmax(dim=-1)

    def decode(self, ids, scale: torch.Tensor | None = None) -> torch.Tensor:
        """ids: (...,) -> z_hat: (..., D).  Optionally scale by source norm."""
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        z_hat = self.anchors[ids]
        if scale is not None:
            z_hat = z_hat * scale.unsqueeze(-1)
        return z_hat

    @torch.no_grad()
    def roundtrip(self, z: torch.Tensor):
        """Encode/decode and report mean cosine fidelity."""
        ids = self.encode(z)
        z_hat = self.decode(ids, scale=z.norm(dim=-1, keepdim=True))
        cos = F.cosine_similarity(z.float(), z_hat.float(), dim=-1).mean().item()
        return ids, z_hat, cos

    # ── tid string bridge ────────────────────────────────────────────────

    @staticmethod
    def ids_to_text(ids) -> str:
        """ids: iterable of ints -> "tid42 tid671 ..." """
        flat = [int(i) for i in torch.as_tensor(ids).reshape(-1).tolist()]
        return " ".join(f"tid{i}" for i in flat)

    @staticmethod
    def tid_to_int(tid_str: str) -> int:
        return int(tid_str[3:])

    @staticmethod
    def tid_to_ids(tids) -> list[int]:
        return [
            int(str(t)[3:]) for t in tids
            if str(t).startswith("tid") and "\x00" not in str(t)
        ]
