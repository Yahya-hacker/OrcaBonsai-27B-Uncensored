# Running OrcaBonsai on your RTX

Ternary Bonsai 2 27B with **runtime refusal ablation** — the weights are never
modified, so `--alpha` is a per-request knob and `--alpha 0` is bit-identical to the
base model.

Five steps. Only step 4 needs a GPU, and only step 5 needs the kernels.

---

## 0. Check the machine

```bash
python tools/doctor.py
```

Reports bandwidth, PCIe generation, VRAM your display is already holding, and whether
the toolchain can target your card — then your actual resident-context budget and
projected decode rate. It catches the one blocking problem up front: Blackwell is
`sm_120` and needs **CUDA 12.8+** with **PyTorch 2.7+/cu128**.

## 1. Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install transformers          # tokenizer only
```

`pip install -e .` builds the CUDA kernels if a toolkit is present and quietly skips
them if not. Either way the engine runs.

## 2. Prove the engine works — no model download needed

```bash
pytest tests/ -q                  # 164 tests, CPU only
python tools/smoke_test.py        # end-to-end on a miniature real model
python tools/smoke_test.py --cuda # ... and the compiled kernels, with parity
```

`smoke_test.py` builds a structurally identical 4-layer model with real packed weights
in a real container, then runs prefill, incremental decode, generation and the
ablation through the actual code path. It checks that torch dequantisation is
**bit-exact** against the numpy oracle, that incremental decode equals a single
prefill pass (the test that catches nearly every cache/state bug), and that `alpha=0`
is bit-identical.

If this passes, the only thing between you and the 27B model is the weights.

## 3. Get and convert the weights

Download the **PTQ1_0** GGUF from `prism-ml/Ternary-Bonsai-2-27B-gguf` (~5.9 GB).

```bash
python tools/convert.py inspect  Ternary-Bonsai-2-27B-PTQ1_0.gguf
python tools/convert.py convert  Ternary-Bonsai-2-27B-PTQ1_0.gguf  bonsai-27b.bonsai --verify
```

Run `inspect` **first**. It is header-only and takes about a second, and it prints any
ternary tensor whose name the converter does not recognise. The gated-delta layer names
are the one part of the mapping that has not been checked against a real file — if
`inspect` lists unmapped tensors, send me that output and the fix is a few lines in
`tools/convert.py`.

`convert` streams, never holds the whole model in RAM, and `--verify` re-reads every
tensor and compares byte-for-byte. It **refuses** PQ2_0 (1.3 GB larger, which costs you
resident KV on 8 GB) and refuses names it does not recognise rather than guessing.

## 4. Run

```bash
python run_cuda.py --model bonsai-27b.bonsai --prompt "Explain ternary quantisation."
python run_cuda.py --model bonsai-27b.bonsai --chat
```

In chat mode, `/alpha 0.5` retunes the ablation mid-conversation and `/reset` clears
history. Useful knobs:

| flag | meaning |
|---|---|
| `--alpha` | 0 off (bit-identical to base) · ~0.8 partial · 1.0 full · >1 over-projects |
| `--direction none` | disable ablation entirely |
| `--no-kernels` | force the portable torch path — same tokens, lower throughput |
| `--temperature 0` | greedy |

**This works before the kernels compile.** The portable torch path produces identical
output; it is just slower. That is deliberate — it means compiling is a performance
change with a correctness test attached, not the step that decides whether anything
works.

## 5. Compile the kernels for speed

See [`BUILDING-CUDA.md`](BUILDING-CUDA.md). Then:

```bash
python tools/smoke_test.py --cuda     # parity: kernels vs torch
```

---

## Tuning alpha

The direction was estimated on the **bf16 base model**, and whether it transfers
cleanly through quantisation-aware training is unmeasured. Expect to sweep:

```bash
for a in 0 0.6 0.8 1.0 1.2; do
  python run_cuda.py --model bonsai-27b.bonsai --alpha $a --temperature 0 \
      --prompt "<your probe>"
done
```

Too low and refusals survive; too high and the model degrades, because you are
removing a direction that carries some ordinary meaning too. Somewhere around 0.8–1.0
is the usual answer, but measure it.

## What to expect on 8 GB

| | |
|---|---|
| weights (PTQ1_0, 1.75 bpw) | 5.48 GiB |
| GDN state + overhead | ~0.79 GiB |
| left for KV | ~1.34 GiB |
| KV at 4 bits | 19 KiB/token → **~74–82K tokens resident** |
| decode, 32K context | **~28 tok/s** projected at 384 GB/s |

The 4-bit KV cache is what makes this work. At fp16 the cache is 64 KiB/token, so 32K
context needs 2 GiB that you do not have, and spilling it to system RAM caps decode
around 6 tok/s — decode re-reads the whole cache every token.

## Troubleshooting

| symptom | cause |
|---|---|
| `no kernel image is available` | toolkit older than CUDA 12.8 → [`BUILDING-CUDA.md`](BUILDING-CUDA.md) |
| converter rejects a tensor name | expected for the GDN layers; send `inspect` output |
| fluent but wrong output | a ternary/Hadamard bug — run `smoke_test.py`, not your eyes |
| CUDA OOM at long context | lower context, or `--kv-bits 4`; check `doctor.py` for VRAM already in use |
| very slow | kernels not compiled; `run_cuda.py` warns on stderr when it falls back |

**Wrong ternary math produces fluent nonsense, not crashes.** A plausible paragraph is
not evidence of correctness. Trust the oracles.
