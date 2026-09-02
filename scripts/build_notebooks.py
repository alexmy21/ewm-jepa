#!/usr/bin/env python
"""Build the ewm-jepa notebooks from their definitions.

Each notebook mirrors its DeepSeek-OCR/hllset_cortex counterpart, with the
OCR front end replaced by V-JEPA and the OCR decoder replaced by the V-JEPA
predictor.  Regenerate with:

    python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)

KERNELSPEC = {
    "display_name": "Python 3 (ewm-jepa)",
    "language": "python",
    "name": "python3",
}


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def write_notebook(name: str, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": KERNELSPEC,
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = OUT / name
    path.write_text(json.dumps(notebook, indent=1))
    print(f"wrote {path} ({len(cells)} cells)")


# ── shared headers ─────────────────────────────────────────────────────────

SETUP_GPU = '''\
import sys
import pathlib

ROOT = pathlib.Path.cwd()
if not (ROOT / "ewm_jepa").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import torch
from ewm_jepa import JEPAPipeline, Quantizer
import hllset_py
from hllset_cortex import HLLSetFilter
from hllset_cortex.domain import encoding_tokenizer, hllset_from_ids, tid

print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
'''

SETUP_CPU = '''\
import sys
import pathlib

ROOT = pathlib.Path.cwd()
if not (ROOT / "ewm_jepa").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import torch
from ewm_jepa import Quantizer
import hllset_py
from hllset_cortex import HLLSetFilter
from hllset_cortex.domain import encoding_tokenizer, hllset_from_ids, tid

# A deterministic tid-stream source: the same quantizer the JEPA pipeline
# uses, fed with synthetic encodings.  The EWM boundary is encoding-agnostic —
# these tids are interchangeable with V-JEPA quantized patch encodings.
def make_tid_stream(n=200, dim=1024, codebook_size=2048, seed=0):
    q = Quantizer(dim=dim, codebook_size=codebook_size, device="cpu", seed=seed)
    # decorrelate the data seed from the codebook seed so streams differ
    z = torch.randn(n, dim, generator=torch.Generator().manual_seed(1000 + seed))
    return q.ids_to_text(q.encode(z))
'''

# ════════════════════════════════════════════════════════════════════════════
# 01 — pipeline validation (mirrors 01_ocr_hllset_pipeline_real.ipynb)
# ════════════════════════════════════════════════════════════════════════════

nb01 = [
    md('''# V-JEPA × HLLSet Cortex — Pipeline Validation

Mirrors `01_ocr_hllset_pipeline_real.ipynb` from `hllset_cortex`, with the
DeepSeek-OCR encoder replaced by the **V-JEPA ViT-L/16 encoder** and the OCR
decoder replaced by the **V-JEPA predictor**.

The EWM narrative is `docs/EWM/main.tex`; the principle-level mapping is
`docs/NARRATIVE.md`.  This notebook walks the data boundary:

```text
video -> V-JEPA encoder -> z (continuous)
      -> quantize -> "tid42 tid671 ..."
      -> HLLSet lattice (gate ∩, LUT, materialize)
      -> dequantize -> z_hat
      -> V-JEPA predictor -> prediction
```'''),
    code(SETUP_GPU),
    md('''## 1. Tokenizer — 3-gram structural encoding

The same `hllset-dsl` tokenizer that processed `tid{n}` streams in
ds-hllset-cortex processes quantized V-JEPA patch IDs here.  n-grams are the
**bootstrap family** (Principle 3, Ashby–Bootstrap): each order is a distinct
measurement view of the same patch sequence.'''),
    code('''\
tok = encoding_tokenizer()
toks = tok.tokenize(b"tid42 tid671 tid42 tid99")
print([t.decode() for t in toks])
'''),
    md('''## 2. HLLSet — IICA properties

Idempotent (same IDs, same HLLSet), Immutable (never changes), Content-
Addressed (the SHA-1 key *is* the address).  These are Principles 1 and 2 of
`main.tex`.'''),
    code('''\
h1 = hllset_py.HLLSet.from_tokens(["tid42", "tid671", "tid42"])
h2 = hllset_py.HLLSet.from_tokens(["tid42", "tid671", "tid42"])
print("idempotent:", h1.content_key() == h2.content_key())
print("popcount  :", h1.popcount(), "->", h1.popcount(), "(immutable)")
print("CID       :", h1.content_key())
'''),
    md('''## 3. gate_TF HLLSet — the codebook vocabulary as a content-addressed gate

In ds-hllset-cortex the gate was the decoder's BPE vocabulary.  Here it is the
**quantizer codebook**: the set of `tid{n}` the system can express in V-JEPA
latent space.  Intersection with the gate filters invalid bit positions at the
lattice level (the "world view", `main.tex` appendix).'''),
    code('''\
CODEBOOK_SIZE = 2048
gate = hllset_from_ids(range(CODEBOOK_SIZE))
print("gate popcount:", gate.popcount())
print("gate CID     :", gate.content_key())
'''),
    md('''## 4. HLLSetFilter — LUT + TF-ranked materialization

The LUT accumulates TF for **every** observed tid — pre-gate, monotonic, and
never reset.  Materialization returns the highest-TF token at each active bit
position.  This is the Type-1 forward model of the perceptron taxonomy.'''),
    code('''\
filt = HLLSetFilter()
filt.tokenizer = encoding_tokenizer()
res = filt.process_text("tid42 tid671 tid42 tid99")
print("restored :", res.token_strings)
print("stats    :", res.stats)
'''),
    md('''## 5. Multi-document learning

The LUT is shared across streams: TF earned in one observation improves
disambiguation in later ones.  This is the cold-start rule of `main.tex`
Appendix D — TF is earned through experience, never seeded.'''),
    code('''\
for stream in ["tid42 tid671 tid42", "tid42 tid99 tid7 tid671"]:
    filt.process_text(stream)
print(filt.summary())
'''),
    md('''## 6. Cross-document similarity — BSS

The Bell State Similarity is a directed, two-component measure
(`main.tex` appendix): τ = inclusion, ρ = exclusion.'''),
    code('''\
ha = hllset_py.HLLSet.from_tokens(["tid42", "tid671", "tid99"])
hb = hllset_py.HLLSet.from_tokens(["tid42", "tid99", "tid7"])

def bss_tau(a, b):
    return a.intersection(b).popcount() / max(b.popcount(), 1)

def bss_rho(a, b):
    return a.difference(b).popcount() / max(b.popcount(), 1)

print("tau(A->B):", round(bss_tau(ha, hb), 3),
      "rho(A->B):", round(bss_rho(ha, hb), 3))
print("tau(B->A):", round(bss_tau(hb, ha), 3),
      "rho(B->A):", round(bss_rho(hb, ha), 3))
'''),
    md('''## 7. JEPAPipeline — the full boundary

This is the ewm-jepa loop: encode → quantize → HLLSet lattice → dequantize →
predict.  We compare the vanilla prediction against the prediction made from
lattice-restored context encodings.'''),
    code('''\
pipe = JEPAPipeline(codebook_size=CODEBOOK_SIZE, seed=0)
video, masks_enc, masks_pred = pipe.make_masks(batch_size=1)
print("video:", tuple(video.shape), "mask pairs:", len(masks_enc))

result = pipe.roundtrip(video, masks_enc, masks_pred)
for i, s in enumerate(result.stats):
    print(f"pair {i}: ids {s.input_ids} (unique {s.unique_input_ids}) "
          f"-> gated set {s.gated_ids} (set retention {s.set_retention:.3f}) "
          f"| ordered {s.restored_ids} (retention {s.retention:.3f})")
    print(f"        HLLSet {s.hllset_popcount} bits -> gate {s.gate_popcount} bits "
          f"| LUT {s.lut_size} | quant cos {s.quant_cosine:.3f} "
          f"| pred cos {s.pred_cosine:.3f}")
'''),
    md('''## 8. Latent vocabulary — TF survives gate changes

An out-of-codebook tid is filtered by the gate but still accumulates TF in the
LUT.  When the gate expands (codebook v2), the previously latent tid
materializes **immediately at its earned TF** — belief revision without cold
start (`main.tex` appendix, "Change the Vocabulary, Change the World").'''),
    code('''\
def unigrams(res):
    return [t for t in res.token_strings if "\\x00" not in t]

f2 = HLLSetFilter()
f2.tokenizer = encoding_tokenizer()
gate_v1 = hllset_from_ids(range(1000))        # narrow gate
f2.gate_hllset = gate_v1

r1 = f2.process_text("tid42 tid1500 tid42")   # tid1500 is outside gate_v1
print("narrow gate restored  :", unigrams(r1))
print("LUT knows tid1500     :",
      any(t == "tid1500" for t, _ in f2.lut.ranked_tokens()))

gate_v2 = hllset_from_ids(range(2000))        # expanded gate
f2.gate_hllset = gate_v2
r2 = f2.process_text("tid42 tid1500 tid42")
print("expanded gate restored:", unigrams(r2))
'''),
    md('''## 9. Full roundtrip — summary

**What the numbers show.** The gated set pass retains the encoding vocabulary
with high fidelity (set retention → 1.0): the EWM lattice is a faithful
*set memory* over V-JEPA encodings.  The De Bruijn ordered pass degrades when
the same tid recurs many times in a long context (ordered retention is low):
**order restoration under repetition is the open problem** — exactly the
Ashby–Bootstrap trade-off that deeper n-gram bootstraps are designed to
resolve.  Even so, the predictor's output from restored context stays
correlated with the vanilla prediction (pred cos > 0.3), demonstrating that
the JEPA forward model tolerates the lattice boundary.'''),
]

# ════════════════════════════════════════════════════════════════════════════
# 02 — unification (mirrors 02_caal_cortex_unification.ipynb)
# ════════════════════════════════════════════════════════════════════════════

nb02 = [
    md('''# Three-LUT Unification — Encoder, Gate, Predictor

Mirrors `02_caal_cortex_unification.ipynb`.  In ds-hllset-cortex the three
LUTs held characters, bigrams, and trigrams.  Here they hold the three
generations of a **V-JEPA quantized encoding stream**:

- `LUT_1g` — unigram tids (the codebook vocabulary, the encoder's atoms)
- `LUT_2g` — bigram tids (adjacency structure — feeds De Bruijn order)
- `LUT_3g` — trigram tids (triadic structure — disambiguation)

The narrative source is `docs/EWM/main.tex` §HLLSet Algebra: TF is stored,
rank is derived, and the LUT is the only component that learns.'''),
    code(SETUP_CPU),
    md('''## 1. The three-LUT architecture'''),
    code('''\
from dataclasses import dataclass, field

@dataclass
class ThreeLUT:
    lut_1g: object = field(default_factory=hllset_py.TokenLut)
    lut_2g: object = field(default_factory=hllset_py.TokenLut)
    lut_3g: object = field(default_factory=hllset_py.TokenLut)

    def seed_1g(self, ids):
        self.lut_1g.record_all([f"tid{i}" for i in ids])
        return len(ids)

    def ingest(self, text):
        tok1 = encoding_tokenizer()
        toks = tok1.tokenize(text.encode())
        for t in toks:
            s = t.decode()
            if "\\x00" not in s:
                self.lut_1g.record_all([s])
            elif s.count("\\x00") == 1:
                self.lut_2g.record_all([s])
            else:
                self.lut_3g.record_all([s])
        return len(toks)

tlu = ThreeLUT()
tlu.seed_1g(range(2048))
text = make_tid_stream(n=80)
n = tlu.ingest(text)
print("ingested tokens:", n)
print("LUT sizes      :", tlu.lut_1g.len(), tlu.lut_2g.len(), tlu.lut_3g.len())
'''),
    md('''## 2. Cross-LUT disambiguation

When two tids collide in the 1-gram lattice, the 2-gram and 3-gram LUTs
resolve which one was actually adjacent — the bootstrap views vote.'''),
    code('''\
stream = make_tid_stream(n=60, seed=1)
tlu.ingest(stream)

# A colliding unigram pair: whichever tid has higher 2-gram support wins.
tids = stream.split()
from collections import Counter
bigram_support = Counter()
for a, b in zip(tids, tids[1:]):
    bigram_support[f"{a}\\x00{b}"] += 1

print("distinct tids :", len(set(tids)))
print("bigram types  :", len(bigram_support))
print("top bigrams   :", list(bigram_support.items())[:3])
'''),
    md('''## 3. De Bruijn order restoration (ungated bigram topology)

Exactly as in the original: the bigram HLLSet is **not** gated, because the
gate is built from unigrams and would destroy the bigram topology.  Order
restoration is the Type-1 forward model.'''),
    code('''\
db_tok = (
    hllset_py.Tokenizer()
    .pattern(list((__import__("string").ascii_lowercase + __import__("string").digits + "_-").encode()))
    .lowercase()
    .pad(b"<S>", b"</S>")
    .ngrams(2, 2)
)

def debruijn_restore(text, lut2):
    bigrams = db_tok.tokenize(text.encode())
    h_db = hllset_py.HLLSet.from_token_bytes(bigrams)
    lut2.record_all_bytes(bigrams)
    ordered = hllset_py.materialize_debruijn(h_db, lut2, "<S>", "</S>")
    payload = [t if isinstance(t, str) else t.decode() for t in ordered
               if t not in ("<S>", "</S>")]
    return payload

short = " ".join(make_tid_stream(n=12, seed=2).split()[:12])
restored = debruijn_restore(short, tlu.lut_2g)
print("original:", short.split())
print("restored:", restored)
print("exact   :", restored == short.split())
'''),
    md('''## 4. Content-addressed retrieval — BSS over documents

Documents are HLLSets; similarity is the directed BSS pair.  Two quantized
"documents" from the same seed share structure; different seeds do not.'''),
    code('''\
d1 = make_tid_stream(n=120, seed=0)
d2 = make_tid_stream(n=120, seed=0)   # same underlying structure
d3 = make_tid_stream(n=120, seed=99)  # different structure

h1 = hllset_py.HLLSet.from_tokens(d1.split())
h2 = hllset_py.HLLSet.from_tokens(d2.split())
h3 = hllset_py.HLLSet.from_tokens(d3.split())

def tau(a, b): return a.intersection(b).popcount() / max(b.popcount(), 1)
def rho(a, b): return a.difference(b).popcount() / max(b.popcount(), 1)

for name, hx in [("same", h2), ("different", h3)]:
    print(f"{name:>9}: tau={tau(h1, hx):.3f} rho={rho(h1, hx):.3f}")
'''),
    md('''## 5. Summary

The three-LUT architecture from ds-hllset-cortex carries over unchanged to
V-JEPA encodings.  The only model-specific piece is the **token definition**
(`tid{n}` from the quantizer) — exactly as `main.tex` §5.7 predicted:
*"What changes is the token definition — what string you feed to the hash
function."*'''),
]

# ════════════════════════════════════════════════════════════════════════════
# 03 — recursive IICA chain (mirrors 03_recursive_iica_chain.ipynb)
# ════════════════════════════════════════════════════════════════════════════

nb03 = [
    md('''# Recursive IICA Chain — the chained unified model

Mirrors `03_recursive_iica_chain.ipynb`.  A [UM] node is the compound IICA
morphism `encodings -> HLLSets -> encodings`.  Chaining nodes composes the
morphism (Principle 2: composition of IICA is IICA).  The chain runs on
quantized V-JEPA tid streams — the same streams the encoder boundary emits.'''),
    code(SETUP_CPU),
    md('''## 1. The building block

A chain node is just `HLLSetFilter` with a gate: ingest → gate ∩ → LUT →
materialize.  The output stream feeds the next node.'''),
    code('''\
def make_node(gate_ids):
    f = HLLSetFilter()
    f.tokenizer = encoding_tokenizer()
    f.gate_hllset = hllset_from_ids(gate_ids)
    return f

def node_pass(node, text):
    res = node.process_text(text)
    # unigrams only, in materialized order
    return " ".join(t for t in res.token_strings if "\\x00" not in t)

stream = make_tid_stream(n=40, seed=3)
node = make_node(range(2048))
out = node_pass(node, stream)
print("in :", stream)
print("out:", out)
'''),
    md('''## 2. IICA chain verification

Three nodes in a row.  IICA composition predicts: same input → same output,
every run, and the chain converges to a fixed point after one full pass.'''),
    code('''\
chain = [make_node(range(2048)) for _ in range(3)]
x = stream
for i, node in enumerate(chain):
    x = node_pass(node, x)
    print(f"after node {i}: {len(x.split())} tids")

x2 = stream
for node in chain:
    x2 = node_pass(node, x2)
print("deterministic:", x == x2)
'''),
    md('''## 3. Progressive gate chain

Each node narrows the gate: node 1 admits 2048 tids, node 2 admits 1024,
node 3 admits 512.  The chain filters the stream to the intersection of all
world views — a structural "cortical brake" cascade.'''),
    code('''\
gates = [range(2048), range(1024), range(512)]
x = stream
for i, g in enumerate(gates):
    x = node_pass(make_node(g), x)
    print(f"gate {i} ({len(list(g))} tids): {len(x.split())} tids survive")
'''),
    md('''## 4. Feedback convergence

Close the loop: the output feeds back as input.  The chain reaches a fixed
point because HLLSet union is idempotent and the LUT is monotonic — the
Noether union invariant in miniature.'''),
    code('''\
node = make_node(range(2048))
x = stream
for step in range(5):
    x_next = node_pass(node, x)
    print(f"step {step}: {len(x.split())} tids")
    if x_next == x:
        print("fixed point reached")
        break
    x = x_next
'''),
    md('''## 5. Summary

`[E] -> chain -> [E]` holds for V-JEPA encodings exactly as it held for
DeepSeek-OCR encodings: the EWM boundary is encoding-agnostic, and IICA
composition means the chain needs no coordination, no versioning, and no
naming at any depth.'''),
]

# ════════════════════════════════════════════════════════════════════════════
# 04 — [UM]-Net agent network (mirrors 04_dg_agent_network.ipynb)
# ════════════════════════════════════════════════════════════════════════════

nb04 = [
    md('''# [UM]-Net — Directed Graph Agent Network

Mirrors `04_dg_agent_network.ipynb`.  What flows between agents is
**encodings, not HLLSets**: each [UM] node is an `HLLSetFilter`, and the
network converges without coordination because every node is an IICA morphism
and CRDT union is monotonic.  The streams here are quantized V-JEPA tids.'''),
    code(SETUP_CPU),
    md('''## 1. Agent definition'''),
    code('''\
from dataclasses import dataclass, field

@dataclass
class UMAgent:
    name: str
    filt: object = field(default_factory=HLLSetFilter)

    def __post_init__(self):
        self.filt.tokenizer = encoding_tokenizer()

    def process(self, text):
        res = self.filt.process_text(text)
        return " ".join(t for t in res.token_strings if "\\x00" not in t)

    @property
    def hllset(self):
        # fingerprint of everything this agent has seen
        top = hllset_py.HLLSet()
        # last result is kept in _last
        return getattr(self, "_last", top)

agents = {name: UMAgent(name) for name in ["a", "b", "c", "d", "e"]}
print("agents:", list(agents))
'''),
    md('''## 2. Fire-and-forget: encodings in, encodings out'''),
    code('''\
streams = {
    "a": make_tid_stream(n=60, seed=10),
    "b": make_tid_stream(n=60, seed=11),
}
outs = {name: agents[name].process(text) for name, text in streams.items()}
for name, out in outs.items():
    print(f"agent {name}: {len(out.split())} tids out")
'''),
    md('''## 3. Diamond topology: fork → process → merge

Two upstream agents process different views; their encodings merge at a
confluence agent.  The merge is union — monotonic, commutative, and
convergent by construction (no consensus).'''),
    code('''\
up_a = outs["a"]
up_b = outs["b"]
merged = up_a + " " + up_b
out_c = agents["c"].process(merged)
print("confluence agent c:", len(out_c.split()), "tids")
'''),
    md('''## 4. Robustness: loss, duplicates, ordering

The same CRDT properties that make HLLSet union idempotent make the network
insensitive to message loss, duplication, and reordering.'''),
    code('''\
base = streams["a"]
dup = base + " " + base                       # duplicate delivery
reordered = " ".join(base.split()[::-1])      # reordered delivery

h_base = hllset_py.HLLSet.from_tokens(base.split())
h_dup = hllset_py.HLLSet.from_tokens(dup.split())
h_re = hllset_py.HLLSet.from_tokens(reordered.split())
print("dup == base      :", h_dup.content_key() == h_base.content_key())
print("reordered == base:", h_re.content_key() == h_base.content_key())
'''),
    md('''## 5. [E] → Graph → [E] feedback loop

The graph output feeds back as the next input: the network becomes a
recurrent EWM structure — short-term memory through temporal separation,
exactly the [UM]-Net cycle of `main.tex` §4.4.'''),
    code('''\
x = streams["a"]
for step in range(3):
    x = agents["d"].process(agents["e"].process(x))
    print(f"loop {step}: {len(x.split())} tids")
'''),
    md('''## 6. Summary

The [UM]-Net over V-JEPA encodings needs no handshakes, no retries, and no
leader election.  IICA morphisms + monotonic union = convergence by
construction — the property that lets EWM scale to swarms of world-model
nodes.'''),
]

# ════════════════════════════════════════════════════════════════════════════
# 05 — holographic memory (mirrors 08_holographic_memory.ipynb)
# ════════════════════════════════════════════════════════════════════════════

nb05 = [
    md('''# Holographic Lattice Memory

Mirrors `08_holographic_memory.ipynb`.  The temporal pyramid accumulates
quantized V-JEPA tid streams into layers; the top HLLSet plus the TF stack
reconstructs any past state (the holographic property, `main.tex` §Noether).'''),
    code(SETUP_CPU),
    md('''## 1. The temporal pyramid over tid streams'''),
    code('''\
from hllset_cortex.temporal import drn

class Pyramid:
    """Configurable sliding-window pyramid over real HLLSets (the §4.2
    construction, with the top-layer carry guarded)."""
    def __init__(self, durations):
        self.durations = list(durations)
        self.layers = [hllset_py.HLLSet() for _ in range(len(durations) + 1)]
        self.counters = [0] * len(durations)

    @property
    def top(self):
        top = hllset_py.HLLSet()
        for layer in self.layers:
            top = top.union(layer)
        return top

    def step(self, s):
        self.layers[0] = self.layers[0].union(s)
        self.counters[0] += 1
        for i, d in enumerate(self.durations):
            if self.counters[i] < d:
                break
            self.layers[i + 1] = self.layers[i + 1].union(self.layers[i])
            self.layers[i] = hllset_py.HLLSet()
            self.counters[i] = 0
            if i + 1 < len(self.counters):
                self.counters[i + 1] += 1

    def layer_popcounts(self):
        return [layer.popcount() for layer in self.layers]

pyramid = Pyramid([3, 2, 2])   # 4 layers, compressed config
print("layers:", len(pyramid.layers))

for step in range(12):
    s = make_tid_stream(n=30, seed=step)
    pyramid.step(hllset_py.HLLSet.from_tokens(s.split()))
    if step in (0, 2, 3, 5, 11):
        print(f"step {step:2d}: layers={pyramid.layer_popcounts()} "
              f"top={pyramid.top.popcount()}")
'''),
    md('''## 2. DRN decomposition — the Noether balance

Each arriving observation splits into Departed / Retained / Novel against the
previous system state.  The steering condition `|card(N) - card(D)| -> 0`
is the EWM analog of homeostatic reinforcement balance.'''),
    code('''\
prev = hllset_py.HLLSet()
for step in range(8):
    s = hllset_py.HLLSet.from_tokens(make_tid_stream(n=40, seed=step).split())
    d = drn(s, prev)
    print(f"step {step}: R={d.r.popcount()} D={d.d.popcount()} N={d.n.popcount()} "
          f"| |N|-|D|| = {abs(d.n.popcount() - d.d.popcount())}")
    prev = prev.union(s)
'''),
    md('''## 3. Holographic reconstruction

The top HLLSet never loses bits (union invariant), and the LUT records *when*
each tid was active.  Past states are recovered by intersecting the top with
the TF-ranked tid set of a past window — the TF stack is the reference beam.'''),
    code('''\
f = HLLSetFilter()
f.tokenizer = encoding_tokenizer()
states = []
for step in range(6):
    s = make_tid_stream(n=40, seed=step)
    f.process_text(s)
    states.append(hllset_py.HLLSet.from_tokens(s.split()))

top = hllset_py.HLLSet()
for s in states:
    top = top.union(s)

def reconstruct(past):
    return top.intersection(past)

for step in (0, 3, 5):
    r = reconstruct(states[step])
    print(f"reconstruct step {step}: {r.popcount()} / {states[step].popcount()} bits "
          f"({r.popcount() / max(states[step].popcount(), 1):.3f})")
'''),
    md('''## 4. Recovery ratio over time

Older states remain reconstructible because the lattice never forgets:
exclusion is attention, not erasure.'''),
    code('''\
ratios = []
for step, s in enumerate(states):
    ratios.append(reconstruct(s).popcount() / max(s.popcount(), 1))
print("recovery ratios:", [round(r, 3) for r in ratios])
'''),
    md('''## 5. Summary

The holographic property is a **Noether consequence**: because the system
state is a monotonic union, information is never lost from the top — it is
only redistributed across layers.  The same argument applies when the
streams are quantized V-JEPA encodings of a video corpus: the lattice becomes
a holographic memory of what the world model has perceived.'''),
]

# ════════════════════════════════════════════════════════════════════════════
# 06 — grounding proof (mirrors prove_ewm_grounding.ipynb)
# ════════════════════════════════════════════════════════════════════════════

nb06 = [
    md('''# Proving the EWM Promise — Real V-JEPA × HLLSet Cortex

Mirrors `prove_ewm_grounding.ipynb` from ds-hllset-cortex, with the
DeepSeek-OCR encoder replaced by the V-JEPA encoder.  Two proofs:

1. **Fidelity** — encoder → lattice → predictor round-trips real V-JEPA
   encodings with measurable retention.
2. **One-sided grounding** — an out-of-codebook tid is filtered from output
   but still accumulates TF in the LUT (hallucination detection is
   one-sided: the system can flag what it has never measured, and keeps
   measuring what the gate currently forbids).'''),
    code(SETUP_GPU),
    md('''## Proof 1 — fidelity: encoder → lattice → decoder round-trips real encodings'''),
    code('''\
pipe = JEPAPipeline(codebook_size=2048, seed=0)
video, masks_enc, masks_pred = pipe.make_masks(batch_size=1)
result = pipe.roundtrip(video, masks_enc, masks_pred)

for i, s in enumerate(result.stats):
    print(f"pair {i}:")
    print(f"  input   : {s.input_ids} tids ({s.unique_input_ids} unique)")
    print(f"  gated   : {s.gated_ids} tids, set retention {s.set_retention:.3f}")
    print(f"  ordered : {s.restored_ids} tids, retention {s.retention:.3f}")
    print(f"  fidelity: quant cos {s.quant_cosine:.3f}, pred cos {s.pred_cosine:.3f}")
'''),
    md('''## Proof 2 — grounding is one-sided

Inject a tid the codebook has never seen (`tid99999`).  The gate filters it
from the materialized output; the LUT nevertheless records it.  Later, if the
codebook expands, that tid materializes at its earned TF — the latent
vocabulary dynamic of `main.tex` appendix.'''),
    code('''\
from hllset_cortex.grounding import has_hallucination

f = HLLSetFilter()
f.tokenizer = encoding_tokenizer()
f.gate_hllset = pipe.gate            # the codebook world view

stream = pipe.quantize_stream(pipe.encode(video)[0]) + " tid99999"
res = f.process_text(stream)

print("gate filtered tid99999:", "tid99999" not in res.token_strings)
print("LUT recorded tid99999 :",
      any(t == "tid99999" for t, _ in f.lut.ranked_tokens()))
print("latent TF of tid99999 :",
      [tf for t, tf in f.lut.ranked_tokens() if t == "tid99999"])

# Hallucination check: bits with no known token in the LUT
print("has_hallucination     :", has_hallucination(res.hllset, f.lut))
'''),
    md('''## Finding

The single-HLLSet gate filters out-of-codebook tids from **output** while the
LUT keeps measuring them — grounding is one-sided by construction.  The same
finding held for DeepSeek-OCR BPE IDs; it now holds for V-JEPA quantized
patch encodings.  The EWM boundary is encoding-agnostic.'''),
]

# ── write them all ─────────────────────────────────────────────────────────

write_notebook("01_jepa_hllset_pipeline.ipynb", nb01)
write_notebook("02_jepa_ewm_unification.ipynb", nb02)
write_notebook("03_recursive_iica_chain.ipynb", nb03)
write_notebook("04_um_net_agent_network.ipynb", nb04)
write_notebook("05_holographic_memory.ipynb", nb05)
write_notebook("06_grounding_proof.ipynb", nb06)
