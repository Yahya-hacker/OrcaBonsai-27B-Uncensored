# Native CUDA runtime — status and quickstart

A ground-up CUDA/PyTorch runtime for Ternary Bonsai 2 27B with the refusal ablation
fused into the engine, targeting a **fluent agentic experience on an 8 GB RTX**.

Design rationale, budgets and the full risk register live in
[`PLAN-NATIVE-CUDA.md`](../PLAN-NATIVE-CUDA.md). This file is what to actually run.

---

## Where it stands

| Milestone | State |
|---|---|
| **M0** converter, container, oracles | **in progress — codec, container and converter done and tested** |
| M1 GEMV + FWHT kernels | reference CUDA written, never compiled |
| M2 decode path + 4-bit KV | not started |
| M3 fused ablation | reference CUDA written, semantics verified on CPU |
| M4 prefill + prefix cache | not started |
| M5–M6 perf, packaging | not started |

**56 tests pass on CPU with numpy alone.** No GPU was available while authoring, so
nothing in `bonsai/kernels/` has been compiled. Everything that *could* be verified
without a GPU has been, including the kernels' index and bit arithmetic — see
`tests/test_kernel_semantics.py`, which reimplements the device functions in numpy and
checks them against the reference codec.

---

## Step 1 — tell us about your machine

Two commands, one minute, no downloads. They resolve the last open variables in the
memory budget (§0 of the plan): real memory bandwidth, PCIe generation, and how much
VRAM your display is already holding.

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,pcie.link.gen.current,pcie.link.width.current --format=csv
nvidia-smi --query-gpu=clocks.max.memory --format=csv
python -c "import torch;print(torch.__version__, torch.cuda.get_device_capability())"
```

Expected on the target: `(12, 0)` — Blackwell sm_120, which needs **CUDA 12.8+** and
**PyTorch 2.7+/cu128**. An older stack fails at *runtime* with `no kernel image is
available for execution on the device`, not at build time (R12).

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
pytest tests/ -q          # 56 tests, CPU only, no model files needed
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
