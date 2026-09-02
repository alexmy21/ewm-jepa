# ewm_jepa
"""EWM × V-JEPA: the HLLSet lattice between the V-JEPA encoder and predictor.

See docs/NARRATIVE.md for the principle-level mapping and docs/EWM/main.tex
for the canonical EWM consolidation document.
"""

from .quantize import Quantizer
from .vjepa import (
    load_vjepa,
    make_mask_collator,
    random_video_and_masks,
    encode_full,
    encode_masked,
    target_encodings,
    predict,
)
from .pipeline import (
    JEPAPipeline,
    RoundTripResult,
    RoundTripStats,
)

__all__ = [
    "Quantizer",
    "load_vjepa",
    "make_mask_collator",
    "random_video_and_masks",
    "encode_full",
    "encode_masked",
    "target_encodings",
    "predict",
    "JEPAPipeline",
    "RoundTripResult",
    "RoundTripStats",
]
