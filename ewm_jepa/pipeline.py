# ewm_jepa/pipeline.py
"""JEPAPipeline — the EWM lattice between the V-JEPA encoder and predictor.

This is the black-box bridge, mirroring ``hllset_cortex.pipeline.OCRPipeline``
but with the OCR front end replaced by V-JEPA:

    video -> V-JEPA encoder -> z (continuous)          \
    z -> Quantizer -> "tid42 tid671 ..."                \
    tids -> HLLSetFilter (gate ∩, LUT, materialize)      -> V-JEPA predictor
    restored tids -> Quantizer.decode -> z_hat -------->

The gate_TF HLLSet is built once from the quantizer codebook (the set of
``tid{n}`` the system can express in its latent space).  The LUT accumulates
TF for every observed ``tid`` — monotonic, pre-gate, exactly as in
ds-hllset-cortex.

Two passes happen inside the lattice, mirroring hllset_cortex:

1. **Gated set pass** — the tid stream is tokenized (1-, 2-, 3-grams),
   fingerprinted into an HLLSet, intersected with the gate, recorded into the
   LUT, and materialized back to the valid tid set (TF-ranked).
2. **Ungated De Bruijn pass** — NUL-separated bigrams keep the sequence
   topology; ``materialize_debruijn`` restores order.  The bigram HLLSet is
   deliberately *not* gated (gating would destroy the bigram topology), as in
   the original hllset_cortex notebook 02.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

import hllset_py
from hllset_cortex import HLLSetFilter
from hllset_cortex.domain import encoding_tokenizer, hllset_from_ids

from .quantize import Quantizer
from .vjepa import (
    encode_full,
    encode_masked,
    load_vjepa,
    predict,
    random_video_and_masks,
    target_encodings,
)


def _unfold_debruijn_path(tokens: List[str]) -> List[str]:
    """Flatten a De Bruijn path that mixes node tokens and NUL edge tokens.

    ``materialize_debruijn`` may return edge labels (``a\\x00b``) when the
    graph branches; unfold them into node sequence, deduplicating joints.
    """
    seq: List[str] = []
    for t in tokens:
        if "\x00" in t:
            a, b = t.split("\x00", 1)
            for x in (a, b):
                if not seq or seq[-1] != x:
                    seq.append(x)
        else:
            if not seq or seq[-1] != t:
                seq.append(t)
    return seq


@dataclass
class CortexPassResult:
    """One lattice pass over a tid stream."""

    gated_tokens: List[str] = field(default_factory=list)
    ordered_tokens: List[str] = field(default_factory=list)
    hllset: Optional[object] = None
    filtered_hllset: Optional[object] = None
    db_hllset: Optional[object] = None
    lut_size: int = 0


@dataclass
class RoundTripStats:
    """One lattice round-trip of a context encoding stream."""

    input_ids: int = 0
    unique_input_ids: int = 0
    gated_ids: int = 0
    restored_ids: int = 0
    retention: float = 0.0
    set_retention: float = 0.0
    hllset_popcount: int = 0
    gate_popcount: int = 0
    lut_size: int = 0
    quant_cosine: float = 0.0
    pred_cosine: float = 0.0


@dataclass
class RoundTripResult:
    pred_original: List[torch.Tensor] = field(default_factory=list)
    pred_restored: List[torch.Tensor] = field(default_factory=list)
    z_restored: List[torch.Tensor] = field(default_factory=list)
    stats: List[RoundTripStats] = field(default_factory=list)


@dataclass
class JEPAPipeline:
    """V-JEPA + HLLSet lattice, wired as one pipeline.

    Usage:
        pipe = JEPAPipeline()                  # loads ViT-L/16 on cuda:0
        video, masks_enc, masks_pred = pipe.make_masks()
        result = pipe.roundtrip(video, masks_enc, masks_pred)
    """

    device: str = "cuda:0"
    checkpoint: Optional[str] = None
    codebook_size: int = 4096
    num_frames: int = 16
    crop_size: int = 224
    seed: int = 0

    encoder: object = field(init=False, repr=False)
    predictor: object = field(init=False, repr=False)
    quantizer: Quantizer = field(init=False, repr=False)
    filter: HLLSetFilter = field(init=False, repr=False)
    gate: object = field(init=False, repr=False)
    _db_tokenizer: object = field(init=False, repr=False)

    def __post_init__(self):
        self.encoder, self.predictor = load_vjepa(
            device=self.device,
            checkpoint=self.checkpoint,
            num_frames=self.num_frames,
            crop_size=self.crop_size,
        )
        self.quantizer = Quantizer(
            dim=self.encoder.backbone.embed_dim,
            codebook_size=self.codebook_size,
            seed=self.seed,
            device=self.device,
        )
        # EWM side: the same HLLSetFilter used by ds-hllset-cortex, with the
        # encoding-ID tokenizer and a gate built from the codebook vocabulary.
        self.filter = HLLSetFilter()
        self.filter.tokenizer = encoding_tokenizer()
        self.gate = hllset_from_ids(range(self.codebook_size))
        self.filter.gate_hllset = self.gate

        # De Bruijn tokenizer that accepts tid{n} identifiers (the stock
        # debruijn_tokenizer in hllset_cortex only matches a-z words).
        allowed = list((string.ascii_lowercase + string.digits + "_-").encode())
        self._db_tokenizer = (
            hllset_py.Tokenizer()
            .pattern(allowed)
            .lowercase()
            .pad(b"<S>", b"</S>")
            .ngrams(2, 2)
        )

    # ── helpers ──────────────────────────────────────────────────────────

    def make_masks(self, batch_size: int = 1):
        return random_video_and_masks(
            batch_size=batch_size,
            num_frames=self.num_frames,
            crop_size=self.crop_size,
            device=self.device,
            seed=self.seed,
        )

    def encode(self, video):
        return encode_full(self.encoder, video)

    def quantize_stream(self, z: torch.Tensor) -> str:
        """z: (K, D) -> "tid42 tid671 ..." (the measurement stream)."""
        ids = self.quantizer.encode(z)
        return self.quantizer.ids_to_text(ids)

    def cortex_pass(self, text: str) -> CortexPassResult:
        """Encoding-ID text -> HLLSet -> gate ∩ -> LUT -> materialize.

        Runs the gated set pass and the ungated De Bruijn order pass.
        """
        result = CortexPassResult()

        # 1) Gated set pass (1-, 2-, 3-gram tokenizer, gate ∩, LUT, TF-rank).
        fres = self.filter.process_text(text)
        result.hllset = fres.hllset
        result.filtered_hllset = fres.filtered_hllset
        result.lut_size = fres.lut_size
        result.gated_tokens = [
            t for t in fres.token_strings
            if "\x00" not in t and t not in ("<S>", "</S>")
        ]

        # 2) Ungated De Bruijn order pass over NUL-separated bigrams.
        bigrams = self._db_tokenizer.tokenize(text.encode())
        if bigrams:
            h_db = hllset_py.HLLSet.from_token_bytes(bigrams)
            self.filter.lut.record_all_bytes(bigrams)
            ordered = hllset_py.materialize_debruijn(
                h_db, self.filter.lut, "<S>", "</S>"
            )
            result.db_hllset = h_db
            payload = [
                t if isinstance(t, str) else t.decode("utf-8", errors="replace")
                for t in ordered
                if t not in ("<S>", "</S>")
            ]
            result.ordered_tokens = _unfold_debruijn_path(payload)

        return result

    def dequantize_stream(self, restored_tids, k: int) -> torch.Tensor:
        """Restored tid strings -> z_hat (k, D); pads with tid0 if short."""
        ids = self.quantizer.tid_to_ids(restored_tids)
        if len(ids) < k:
            ids += [0] * (k - len(ids))
        elif len(ids) > k:
            ids = ids[:k]
        return self.quantizer.decode(ids)

    def roundtrip(
        self,
        video: torch.Tensor,
        masks_enc: List[torch.Tensor],
        masks_pred: List[torch.Tensor],
    ) -> RoundTripResult:
        """Run the full ewm-jepa loop and compare against the vanilla path.

        Vanilla:   pred = predictor(encoder(video, masks_enc), targets)
        ewm-jepa:  pred = predictor(lattice(encoder(video, masks_enc)), targets)
        """
        with torch.no_grad():
            z_list = encode_masked(self.encoder, video, masks_enc)
            h_list = target_encodings(self.encoder, video, masks_pred)
            pred_original = predict(self.predictor, z_list, h_list, masks_enc, masks_pred)

        result = RoundTripResult(pred_original=pred_original)

        for i, (z, me, mp) in enumerate(zip(z_list, masks_enc, masks_pred)):
            zq = z[0]                      # (K, D)
            ids = self.quantizer.encode(zq)  # (K,)
            text = self.quantizer.ids_to_text(ids)

            cres = self.cortex_pass(text)
            restored_tids = cres.ordered_tokens or cres.gated_tokens

            z_hat = self.dequantize_stream(restored_tids, k=len(ids))
            z_rest = z_hat.unsqueeze(0)     # (1, K, D)

            with torch.no_grad():
                pred_rest = predict(
                    self.predictor,
                    [z_rest],
                    [h_list[i]],
                    [me],
                    [mp],
                )[0]

            stats = RoundTripStats(
                input_ids=len(ids),
                unique_input_ids=int(torch.unique(ids).numel()),
                gated_ids=len(cres.gated_tokens),
                restored_ids=len(restored_tids),
                retention=len(restored_tids) / max(len(ids), 1),
                set_retention=len(cres.gated_tokens) / max(int(torch.unique(ids).numel()), 1),
                hllset_popcount=cres.hllset.popcount() if cres.hllset else 0,
                gate_popcount=cres.filtered_hllset.popcount() if cres.filtered_hllset else 0,
                lut_size=cres.lut_size,
                quant_cosine=F.cosine_similarity(
                    zq.float(), z_hat.float(), dim=-1).mean().item(),
                pred_cosine=F.cosine_similarity(
                    pred_original[i].reshape(-1).float(),
                    pred_rest.reshape(-1).float(),
                    dim=0).item(),
            )

            result.pred_restored.append(pred_rest)
            result.z_restored.append(z_rest)
            result.stats.append(stats)

        return result

    def summary(self) -> dict:
        return self.filter.summary()
