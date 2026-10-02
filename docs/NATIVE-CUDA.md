# Native CUDA runtime — status and quickstart

A ground-up CUDA/PyTorch runtime for Ternary Bonsai 2 27B with the refusal ablation
fused into the engine, targeting a **fluent agentic experience on an 8 GB RTX**.

**To actually run the model, start at [`RUNNING.md`](RUNNING.md).** Toolchain setup and
compile instructions are in [`BUILDING-CUDA.md`](BUILDING-CUDA.md); design rationale,
budgets and the risk register are in [`PLAN-NATIVE-CUDA.md`](../PLAN-NATIVE-CUDA.md).

---

## Where it stands

| Milestone | State |
|---|---|
| **M0** converter, container, oracles | **done** |
| **M1** kernels | reference CUDA + build system written, **never compiled** |
| **M2** decode path + 4-bit KV | **done** — engine runs end to end on the portable torch path |
| **M3** runtime ablation | **done** — policy, engine wiring and CLI |
| M4 prefill + prefix cache | chunked GDN prefill outstanding |
| M5–M6 perf, packaging | not started |

**The engine runs today without any compiled kernel.** `python tools/smoke_test.py`
builds a miniature real model and takes it through prefill, incremental decode,
generation and the ablation. Compiling the kernels is a throughput change with a
parity test attached — not the step that decides whether anything works.
See [`RUNNING.md`](RUNNING.md).

**164 tests pass on CPU** (numpy for the core, torch for the engine). No GPU was available while authoring, so
nothing in `bonsai/kernels/` has been compiled. Everything that *could* be verified
without a GPU has been, including the kernels' index and bit arithmetic — see
`tests/test_kernel_semantics.py`, which reimplements the device functions in numpy and
checks them against the reference codec.

---

## Step 1 — run the doctor

One command, no downloads, safe with or without a GPU. It resolves every variable the
memory budget currently assumes — real bandwidth, PCIe generation, VRAM already held by
the display — and then prints the resident-context budget and projected decode rate
**for your machine** rather than for a datasheet.

```bash
python tools/doctor.py -o doctor.json
```

It also catches the one blocking toolchain problem up front: Blackwell is `sm_120` and
needs **CUDA 12.8+** with **PyTorch 2.7+/cu128**. An older stack fails at *runtime* with
`no kernel image is available for execution on the device`, which never mentions the
real cause (R12). See [`BUILDING-CUDA.md`](BUILDING-CUDA.md).

## Step 2 — get the right GGUF

Download **`*-PTQ1_0.gguf`** (1.76 bpw, 5.93 GB), not PQ2_0.

PQ2_0 unpacks faster, but it is 7.25 GB, and on 8 GB that extra 1.3 GB is the entire
difference between a KV cache that lives in VRAM and one that crawls across PCIe. The
converter **refuses** PQ2_0 rather than silently producing something that will not fit.

## Step 3 — inspect before converting

```bash
python tools/convert.py inspect /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf -o report.json
```

Reads only the header; finishes in about a second on a 5.9 GB file. It prints the
tensor inventory, the ggml type histogram, the architecture metadata, and — the point
of the exercise — **any ternary tensor whose name the converter cannot place**.

The GGUF tensor names for the gated-delta layers are the one thing not yet confirmed
against a real file. If `inspect` reports unmapped tensors, add them to `LAYER_MAP` in
`tools/convert.py`. The converter will not guess: a wrong name map produces a model
that is fluent and wrong, which is the worst failure mode this project has.

## Step 4 — convert

```bash
python tools/convert.py convert /path/to/...-PTQ1_0.gguf -o bonsai-27b.bonsai --verify
```

Streams one tensor at a time, so peak RSS stays in the hundreds of MB. `--verify` reads
every tensor back and asserts the bytes are identical to the source.

---

## What the container does

`.bonsai` is a regrouping of PTQ1_0, not a requantisation — **not one trit changes**.

| | GGUF PTQ1_0 | `.bonsai` |
|---|---|---|
| scales | interleaved, 2 bytes every 28 | separate plane, fully coalesced |
| block | 28 bytes, nothing 16-byte aligned | 1024-weight superblock, 208 B codes, `uint4`-loadable |
| bias | — | dropped: `bias == -scale` exactly, baked into the kernel |
| density | 1.76 bpw | **1.750 bpw exactly** |

Trits are kept in PTQ1_0's native stage order inside each group. A finer interleave may
be faster, but choosing one without a profiler is guesswork, and keeping the bytes
verbatim makes conversion *provably* lossless. `layout_id` is versioned so M1 can change
it once there is hardware to measure on.

## The oracle

The hardest problem in this port is that a 27B hybrid in a bespoke ternary format
yields **fluent nonsense** rather than a crash when something is subtly wrong. The
cheap way out is per-matrix reconstruction:

```python
from bonsai.format import BonsaiReader
r = BonsaiReader("bonsai-27b.bonsai")
W = r.dequantize("model.layers.0.mlp.down_proj")   # exact fp32, 340 MiB, ~1 s
assert allclose(my_kernel(x, packed), x @ W.T)
```

One matrix at a time, exact, no MLX and no llama.cpp required. The largest tensor in
the model is 17408×5120 and reconstructs in about a second.

---

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-native.txt
pytest tests/ -q          # 164 tests, CPU only, no model files needed
```

`bonsai/` (native) and `bonsai_abliterate/` (the original MLX path) coexist. Nothing in
the MLX tree has been modified; it stays as a reference oracle until M2 parity passes.

## Verified architecture facts

`bonsai/config.py` is executable documentation. Its derivation self-checks three ways,
and `tests/` asserts all three:

- **402** packed modules — matches the pack manifest and the GGUF type-142/143 count
- **26.896 B** parameters — matches the published `llama-bench` figure of 26.90 B
- **129** residual writers — 64 `down_proj` + 48 `out_proj` + 16 `o_proj` + `embed_tokens`,
  matching `directions/direction.json` and the 258 tensors in the shipped LoRA

Three traps encoded there rather than left to memory: the doubled `q_proj` whose second
half is a sigmoid output gate; the GDN `inv²` on q against `inv` on k; and the GGUF
value-head permutation, which applies to six tensors but **not** to `ssm_out`.


---

## The KV cache, and why it is the whole project

Decode re-reads the **entire** KV cache every token. At fp16 this model costs 64 KiB
per token across its 16 full-attention layers, so 32K context is 2 GiB — which does not
fit on an 8 GB card beside 5.53 GiB of weights, and therefore has to live in system RAM.
Over PCIe that caps decode at roughly **6 tok/s at 32K** and under **2 tok/s at 262K**,
against ~15 ms to read every weight in the model. KV traffic, not weight traffic, is the
bottleneck.

`bonsai/kvcache.py` removes it:

| | fp16 | this |
|---|---|---|
| bits/value | 16 | **4.75** (K group 32, V group 64, asymmetric) |
| bytes/token | 64 KiB | **19 KiB** |
| resident on 8 GB | ~21K tokens | **~82K tokens** |
| decode at 32K | ~6 tok/s (offloaded) | **~28 tok/s** (resident) |

**K is quantised more finely than V.** K errors perturb attention *logits*, which are
then exponentiated, so they compound; V errors enter the output linearly and partly
average out. Group 32 for K and 64 for V costs 5% more memory than a uniform group-64
cache and measurably protects the logits.

**Group 128 is a trap.** It saves 0.25 bits and loses ~80% more accuracy once outlier
channels are present — and attention K/V genuinely have them. Measured, not assumed.

**~10% elementwise error is inherent to int4**, not a defect: 16 levels across a group's
min-max span puts the RMS error at `span/15/sqrt(12) ≈ 0.11σ`. It is survivable because
softmax is contractive and the most recent 128 tokens stay in fp16 — but it is not
self-evidently fine, and R11 stands: measure perplexity and needle-retrieval on the real
model before trusting it. `kvcache.measure_error()` exists for exactly that.
