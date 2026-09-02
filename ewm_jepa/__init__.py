# ewm_jepa
"""EWM × V-JEPA: the HLLSet lattice between the V-JEPA encoder and predictor.

See docs/NARRATIVE.md for the principle-level mapping and docs/EWM/main.tex
for the canonical EWM consolidation document.
"""

import sys
from pathlib import Path

# Belt-and-suspenders: make the vendored hllset_cortex package importable
# even if the editable install is missing or a stale kernel is running.
_VENDOR_SRC = Path(__file__).resolve().parents[1] / "vendor" / "hllset_cortex" / "src"
if _VENDOR_SRC.exists() and str(_VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(_VENDOR_SRC))

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
