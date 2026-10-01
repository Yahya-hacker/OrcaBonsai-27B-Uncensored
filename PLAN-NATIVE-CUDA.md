# OrcaBonsai → Native CUDA

**Plan only. No existing repository file has been modified.** This document is the sole addition.

Target: **our own CUDA kernels, Python (PyTorch) host.** No MLX. The refusal projection stops being
a wrapper and becomes part of the engine.

Date: 2026-10-01 · Branch `arena/01a0f897-orcabonsai-27b-uncensored`

---

## 0. Target machine — and the one number that redefines the project

**Ryzen 9 8940HX (16C/32T Zen 4) · 32 GB DDR5 · RTX 5060 Laptop GPU, 8 GB GDDR7**
GB206 / GN22-X4, sm_120 Blackwell, 3328 cores, 128-bit bus, **TGP 45–100 W**.
⚠ Two facts to confirm on the machine: sources disagree on memory clock (**384 GB/s** @24 Gbps per
Notebookcheck vs **448 GB/s** @28 Gbps per VideoCardz — laptop SKUs usually take the lower bin), and
the PCIe link width/gen. Both are a one-line measurement and both matter. Plan assumes 384 GB/s.
Also note: NVIDIA ships no "5060 **Ti** Laptop" — the mobile part is the RTX 5060 Laptop GPU.

### The finding

You said KV was offloaded to ~16 GB of system RAM. That is not an incidental detail — it is
*the* bottleneck, and it is worth seeing exactly why:

> **16 GiB ÷ 64 KiB/token = 262,144 tokens.** You were running the full 262K context with the
> entire KV cache in host RAM.

Decode must re-read the **whole** KV cache every single token. Over PCIe that costs:

| Context | KV size | PCIe 4.0 ×8 (~13 GB/s) | PCIe 5.0 ×8 (~25 GB/s) |
|---|---|---|---|
| 8K | 0.50 GiB | 41 ms → **24 tok/s** | 21 ms → **47 tok/s** |
| 32K | 2.00 GiB | 165 ms → **6.1 tok/s** | 86 ms → **11.6 tok/s** |
| 64K | 4.00 GiB | 330 ms → **3.0 tok/s** | 172 ms → **5.8 tok/s** |
| 262K | 16.0 GiB | 1322 ms → **0.8 tok/s** | 687 ms → **1.5 tok/s** |

Reading *all 26.9 B weights* from VRAM costs **~15 ms**. So at 32K the KV transfer is **6–11× more
expensive than the entire rest of the model**, and at 262K it is ~**50×**. No kernel optimisation
anywhere else can matter while this is true. This is why it isn't fluent.

### The fix

Stop offloading. Make the KV cache small enough to stay resident.

| | VRAM | | |
|---|---|---|---|
| PTQ1_0 weights | 5.53 GiB | GDN recurrent state (fp32) | 0.14 GiB |
| CUDA ctx + workspace | ~0.50 GiB | activations / scratch | ~0.15 GiB |
| **fixed subtotal** | **6.32 GiB** | **free for KV (7.6 GiB usable)** | **1.28 GiB** |

| KV precision | per token | max resident context |
|---|---|---|
| fp16 | 64 KiB | 21K |
| int8 | 32 KiB | 42K |
| **int4** | **16 KiB** | **~84K** |

**4-bit KV ⇒ ~84K tokens fully resident, zero PCIe traffic on the decode path.**

Projected decode, everything resident, 55% of roofline (llama.cpp achieves 43–55% on this model):

| | 8K | 32K | 64K |
|---|---|---|---|
| @384 GB/s | 33 tok/s | **31 tok/s** | 29 tok/s |
| @448 GB/s | 39 tok/s | **36 tok/s** | 34 tok/s |

**~30 tok/s at 32K, versus ~6 tok/s offloaded.** A 5× end-to-end win at useful context — and it
comes from a memory-layout decision, not from heroic kernel work.

### What this does to the plan

1. **4-bit KV quantisation moves from "v2 nicety" to a load-bearing M2 deliverable.** Without it
   the project cannot hit its goal on this machine; with it, nothing else needs offloading.
2. **PTQ1_0 (1.75 bpw) is now mandatory, not preferred.** At 2.25 bpw the weights alone are
   7.14 GiB and there is no room for *any* resident KV.
3. **Prefix caching becomes a headline feature, not a nicety.** Agentic turns re-send a long,
   stable prefix; re-prefilling 32K at ~400–900 tok/s costs 35–80 s per turn. Cache it and TTFT
   collapses to the new tokens only. This is the difference between usable and unusable, and it is
   worth more wall-clock than any kernel in §5.
4. **Context ceiling is honest at ~64–80K**, not 262K. Beyond that, page *cold* prefix blocks to
   host RAM with prefetch — never the active window.
5. **Thermals are a real variable.** At 45 W TGP instead of 100 W, expect materially lower sustained
   clocks. Benchmarks must be sustained-rate, not burst.

### Does this justify the native rewrite? Yes — but for a different reason than speed.

On a desktop card with a short context, a native port would roughly tie llama.cpp: both are
bandwidth-bound reading the same 5.94 GB of weights. **On this machine the bottleneck is KV
traffic, not weight traffic**, and that is precisely where a purpose-built implementation wins —
4-bit KV sized for a 16-of-64-layer hybrid, a resident GDN state, prefix reuse, and CUDA-graph
decode. The honest value proposition is *fluency at long context on 8 GB*, plus runtime-adjustable
α that a baked LoRA cannot offer.

---

## 1. Scope decision

| | |
|---|---|
| **We write** | Ternary GEMV + GEMM, fused Hadamard/sign transform, gated-delta-net (decode step + chunked prefill), depthwise causal conv1d, fused residual-add-with-ablation, the weight converter |
| **We borrow** | FlashAttention/SDPA for the 16 full-attention layers, PyTorch for memory/dtypes/host glue, HF `tokenizers`, the pack's `chat_template.jinja` |
| **We delete** | MLX dependency, `mlx-vlm`, `transformers` (text-only), the CPU Docker image |
| **We keep** | `directions/`, `gguf/` (now a *test oracle*, not the product), `swift/`, the Apache-2.0 attribution chain |

"Native" here means **we own the novel math**. It does not mean rewriting attention that Tri Dao
already perfected — spending three weeks on a head_dim-256 flash kernel to match `flash-attn` is
not engineering, it's cosplay. Attention is 16 of 64 layers and is not where this model is unusual.

---

## 2. One correction to what I told you earlier

I said the ablation would fuse into the **matmul epilogue** for free. That was too glib, and the
reason matters for the design.

In a GEMV the 5120 output elements are **split across blocks** (each block owns a slice of N and
streams K). The projection needs `dot(y, r)` over *all* 5120 outputs — a grid-wide reduction. You
cannot do that in a matmul epilogue without a grid sync.

The correct design is better anyway: **fuse the ablation into the residual add**, which has to
touch `x` and `y` and write `h` regardless.

```
h = x + (y − α·(y·r)·r)
```

One block per token, 5120 floats (20 KB) in shared memory, two-stage reduction, single pass.
**Zero extra memory traffic over the residual add you already owe.** 129 sites, genuinely free —
versus 2 extra matmuls per site in the LoRA path and a separate compiled kernel per site in MLX.

That is a real argument for the rewrite, and it survives contact with how GPUs actually work.

---

## 3. Verified architecture spec

Every number below was derived from artifacts, not from prose, and **cross-checks three ways**:
my packed-module count comes to **402** (matches the pack's manifest and the GGUF type-142/143
tensor count) and the parameter total to **26.9 B** (matches the published `llama-bench` line).

### Global

| | |
|---|---|
| Layers | 64 — `full_attention` at `i % 4 == 3` (3,7,…,63), linear elsewhere |
| hidden | 5120 · FFN 17408 · vocab 248320 · ctx 262144 |
| RMSNorm eps | 1e-6 · `tie_word_embeddings: false` |
| Quant | affine, bits 2, group 128, `bias = −scale` exactly → levels `{−s, 0, +s}` |
| Hadamard | block 1024, explicit ±1 signs, folded on the **input** dim; `token_embd` carries the **inverse** |
| Ablation sites | **129** = 64 `down_proj` + 48 `out_proj` + 16 `o_proj` + 1 `embed_tokens` |

### Full-attention layer (×16)

```
q_proj  5120 → 12288   packed   # 24 heads × 256 × 2  — second half is the output GATE
k_proj  5120 →  1024   packed   # 4 KV heads × 256
v_proj  5120 →  1024   packed
q_norm, k_norm = RMSNorm(256)
rope: partial_rotary_factor 0.25 → 64 of 256 dims, theta 1e7, mrope interleaved [11,11,10]
attn = SDPA(q, k, v, scale = 256^-0.5, causal)
out  = o_proj(attn * sigmoid(gate))      ← gate from q_proj's second half
o_proj  6144 → 5120    packed   ← ABLATION SITE
```

### Linear-attention layer (×48) — gated delta net

```
in_proj_qkv 5120 → 10240  packed   # key_dim*2 + value_dim = 2048*2 + 6144
in_proj_z   5120 →  6144  packed
in_proj_b   5120 →    48  NOT packed
in_proj_a   5120 →    48  NOT packed
conv1d: depthwise, groups=10240, kernel 4, no bias, causal   (fp32)
split conv_out → q,k (16 heads × 128), v (48 heads × 128)

inv = 128^-0.5
q = inv² · rms_norm(q, weight=None, 1e-6)      ← note: inv SQUARED on q
k = inv  · rms_norm(k, weight=None, 1e-6)      ← inv on k
beta = sigmoid(b)
g    = exp(−exp(A_log_fp32) · softplus(a + dt_bias))

delta rule over state [48, 128, 128] fp32:
    state = state · g
    kv    = (state · k).sum(-1)
    delta = (v − kv) · beta
    state = state + k ⊗ delta
    y     = (state · q).sum(-1)

out = norm(y, z) where RMSNormGated(h,z) = (silu(z_f32) · rms_norm(h,w,eps)_f32).to(h.dtype)
out_proj 6144 → 5120  packed   ← ABLATION SITE
```

### MLP (×64)
`down_proj(silu(gate_proj(x)) * up_proj(x))`, 5120→17408→5120, all packed.
`down_proj` ← ABLATION SITE.

### Derivation check
402 packed = 64×3 (mlp) + 48×3 (qkv,z,out) + 16×4 (q,k,v,o) + embed + lm_head ✓
26.9 B params = 17.1 (mlp) + 5.53 (linear) + 1.68 (full) + 2.54 (embed+head) ✓

---

## 4. Storage format — the one decision with a hardware payoff

Three candidate sources:

| Source | bpw | LM size | Notes |
|---|---|---|---|
| MLX pack safetensors | 2.25 | 7.67 GB | wastes 0.39 GB on `biases` that are provably `−scales` |
| GGUF **PQ2_0** | 2.16 | 7.25 GB | vendor: "cheaper to unpack, prompt processing is faster" |
| GGUF **PTQ1_0** | **1.76** | **5.93 GB** | 26 B of base-3 trits + 2 B fp16 scale per 128 weights |

(Sizes are PrismML's published figures; my independent arithmetic from 26.90 B params reproduces
them to within a rounding step, which is a useful confirmation of the parameter count.)

### Why not the 3.9 GB model?

PrismML's docs advertise a **27B that fits in 3.9 GB** — but that is **Bonsai 27B (1-bit)**, the
earlier family, weights in {−1,+1} at 1.125 bpw. It is *not* this model. **Ternary Bonsai 2 27B**
is {−1,0,+1} and ships at 5.93 GB minimum. Rejected because:

- **The direction in `directions/` does not transfer.** It was estimated on this model's bf16 base
  (Qwen3.8-27B, hidden 5120) and the 129 sites are this architecture's. A different family needs
  the whole abliteration redone from scratch.
- **Quality.** Bonsai 2 retains 98.2% of its FP16 counterpart and scores 84.78 — against Unsloth's
  UD-Q4_K_XL at 85.18 and 17.6 GB. The 1-bit family is materially weaker.
- 3 GB is not reachable for *any* 27B ternary model regardless: the entropy floor is
  log2(3) = 1.585 bpw → **4.96 GiB** before a single scale is stored. 3 GiB would need 0.96
  bits/weight, below even pure binary.

**Decision: PTQ1_0's 1.76 bpw is MANDATORY on this machine**, re-interleaved into our own layout
by an offline converter. Rationale:

- **It is the only format that leaves room for a resident KV cache.** 8 GiB − 5.53 GiB weights
  − 0.64 GiB fixed overhead = 1.28 GiB for KV. At PQ2_0 (6.75 GiB) that margin is ~0.1 GiB; at the
  MLX pack's 2.25 bpw (7.14 GiB) it is negative. Per §0, resident KV is the whole ballgame.
- Decode is **pure bandwidth** — every token reads every weight. 1.76 vs 2.25 bpw is a **direct
  22% decode speedup** on top of that.
- ⚠ We are knowingly giving up PQ2_0's faster prompt processing. That is the right trade here —
  §0 shows prefill is fixable with a prefix cache, whereas a non-resident KV cache is not fixable
  at all — but it raises the stakes on K2/K5 (§5) and on prefix caching in M4.
- Don't let a storage format dictate a kernel. The on-disk trit order is tuned for CPU SIMD; we
  re-interleave for coalesced 128-bit loads and the mma fragment layout at convert time.
- `bias = −scale` is verified exact across all 402 modules, so we store **scales only**.

Weight-only decode roofline (`bandwidth ÷ 5.94 GB`); §0 has the realistic figures including KV:

| GPU | BW | ceiling |
|---|---|---|
| **RTX 5060 Laptop (target)** | **384–448 GB/s** | **65–75 tok/s** |
| RTX 4090 | 1008 | 170 |
| RTX 3090 | 936 | 158 |

---

## 5. Kernel inventory

**K1 · Ternary GEMV** (decode, M = 1…8) — *the money kernel.*
Bandwidth-bound. 128-bit `uint4` loads of packed trits; decode via a 256-entry shared-memory LUT
(byte → 5 trits) rather than the multiply-shift chain — decode ALU is free when you're memory-bound,
and the LUT keeps registers for accumulators. fp32 accumulate, per-128-group fp16 scale.

**K2 · Ternary GEMM** (prefill, M ≥ 8) — tensor cores.
Decode trits → **fp16** in shared memory, then `mma.m16n8k16.f16.f16.f32`. Double-buffered,
`cp.async` on sm_80+.
*Deliberately not int8.* Int8 activation quantization is the obvious faster path and it is what
mistral.rs chose — but this repo's entire thesis is **"0 additional quantization error."**
Introducing activation quantization to make prefill faster would quietly invalidate the claim the
project exists to make. Revisit only behind an explicit opt-in flag.

**K3 · Fused sign + FWHT-1024.**
`x · signs` then 10 butterfly stages in shared memory (1024 × 4 B = 4 KB), scale 1/32.
**Optimization MLX does not do:** the transform depends only on the *input* vector, but MLX's
`Packed.__call__` recomputes it inside every projection. `gate_proj`/`up_proj` share an input;
`q/k/v_proj` share an input; `in_proj_{qkv,z,a,b}` share an input. Transform **once per distinct
input**: 5 FWHTs per layer instead of 10. Fuse into the preceding RMSNorm's epilogue.

**K4 · GDN decode step** (T=1). State `[48,128,128]` fp32 = 3 MiB/layer, 144 MiB total.
One block per (head, dv-slice); 32 threads across Dk; warp reduce via `__shfl_xor_sync`.

**K5 · GDN chunked prefill** — *the hardest kernel, and the one that decides prefill throughput
for 48 of 64 layers.* Chunkwise parallel delta rule: within-chunk via matmul (tensor cores),
cross-chunk via sequential state passing. Chunk 64 or 128. Budget two weeks for this alone.

**K6 · Depthwise causal conv1d**, 10240 channels, kernel 4, with cache state.

**K7 · Fused residual-add + ablation** (§2). The 129 sites.

**K8 · Misc fused**: RMSNorm(+sign+FWHT), SwiGLU, RMSNormGated (fp32 interior — matters),
partial-RoPE, sampling.

**K9 · 4-bit KV cache + attention over it** — *promoted to critical path by §0.*
Group-wise int4 (group 32–64 along head_dim) with fp16 scale/zero, K and V in paged blocks of 64
or 128 tokens. Two sub-parts: a **quantise-on-append** kernel (cheap, one new token per step), and
**dequantise-in-the-attention-inner-loop**. The second is the constraint on reusing a stock
attention kernel — see below. Only 16 of 64 layers need any of this.
Validate against fp16 KV for perplexity drift; keep the most recent *N* tokens in fp16 (a "sink"
window) if drift shows, which is standard practice and cheap.

**Borrowed, with a caveat:** FlashAttention-2 / `torch.nn.functional.scaled_dot_product_attention`
for the 16 full-attention layers. head_dim 256 is supported on sm_80+, **but**:
- **sm_120 wheel availability is a real risk** — `flash-attn` historically lags new architectures;
  Blackwell consumer needs CUDA 12.8+ and a matching PyTorch (2.7+/cu128). Verify early; fall back
  to PyTorch's cuDNN or mem-efficient SDPA backend.
- **Stock SDPA cannot read a 4-bit KV cache.** So either dequantise a block at a time into a
  scratch buffer before calling SDPA (simple, costs a little extra traffic — still ~10× cheaper
  than PCIe), or write our own decode-attention kernel that dequantises inline. Start with the
  former; the latter is an M5 optimisation. Decode attention over 16 layers is a small fraction of
  the step, so the simple path is likely good enough.

---

## 6. The oracle problem — and why it is solvable cheaply

A 27B hybrid model in a bespoke ternary format **produces fluent nonsense rather than crashing**
when the Hadamard basis or a head permutation is wrong. Without a reference you trust, you will
burn weeks. This is the single biggest risk of the native path.

The unlock is that **you do not need the whole model as an oracle — you need one matrix at a time.**

```python
# Tier 1 oracle: exact, self-contained, no MLX, no llama.cpp.
W_ref = dequantize(codes, scales)          # one matrix, fp16 → 17408×5120 = 178 MB. Trivial.
W_ref = inverse_fwht(W_ref, 1024, signs)   # undo the fold
y_ref = x @ W_ref.T
assert close(our_kernel(x, packed), y_ref)
```

Per-matrix fp16 reconstruction is cheap on any GPU and is **bit-level checkable**. That covers K1,
K2, K3 — the kernels most likely to be subtly wrong — with no external dependency at all.

Five tiers, cheapest first:

| Tier | Oracle | Catches |
|---|---|---|
| 1 | per-matrix fp16 dequant in torch | ternary unpack, scales, Hadamard, layout |
| 2 | pure-torch reference model (slow, fp16, correctness-only) | layer wiring, gating, permutations |
| 3 | `llama.cpp` PrismML fork on CUDA + the LoRA already in `gguf/` | end-to-end logits |
| 4 | this repo's own `selfcheck.py` metric: residual along `r` → ~1e-6 | the ablation itself |
| 5 | the README's eval suite re-run | behaviour |

⚠ **Tier 2 cannot be a materialised fp16 model on this machine.** 26.9 B params in fp16 is 54 GB —
it fits neither in 8 GB of VRAM nor in 32 GB of RAM. Build it instead as a **streaming reference**:
keep the weights packed in VRAM (5.53 GiB) and dequantise **one matrix at a time** into a scratch
buffer (largest is 17408×5120 fp16 = 178 MiB), `torch.matmul`, free. Peak ≈ 5.9 GiB — fits, with
the same weights the fast path uses. It will run at maybe 1–3 tok/s, which is entirely adequate for
correctness work on short prompts. Most testing should be **per-layer** anyway, which needs almost
no VRAM at all.

Tier 2 is still worth building first: it is the regression net for the whole project and it shares
the converter and config with the fast path. **Write it before any CUDA.**

⚠ Tier 3 (`llama.cpp` PrismML fork) is now a **hard prerequisite**, not a convenience — the MLX
oracle is unavailable here, since the 2.25 bpw pack does not fit in 8 GB. Stand llama.cpp up in M0
and keep it as the end-to-end reference for the life of the project.

---

## 7. Risk register — native-specific silent traps

Ordered by how badly they bite. All of these fail *silently*.

### 🔴 R1 · The GDN head permutation
`runtime.py` applies `vperm` when loading **from GGUF**, because `nv=48 ≠ nk=16`:

```python
vperm(unit) = arange(48*unit).reshape(3, 16, unit).transpose(1,0,2).reshape(-1)
```

applied to `attn_qkv` (value part only, after the first `2*nk*hk` rows), `attn_gate`,
`ssm_alpha`, `ssm_beta`, `ssm_a`, `ssm_dt.bias`, and `ssm_conv1d` (value part only).
**The MLX pack already has it applied; the GGUF does not.** And it is *asymmetric* — this repo's
own `export_gguf_lora.py --check` proved `ssm_out`'s input needs **no** permutation (identity
2.08e-04 vs 1.38 under either permutation).
Get this wrong and the model is subtly, fluently wrong.
*Mitigate:* the converter must record its source and assert; port `--check` as a converter test.

### 🔴 R2 · The attention output gate
`q_proj` emits `24 × 256 × 2`. The **second half is a gate**, applied as
`o_proj(attn * sigmoid(gate))`. Miss it and you get a plausible-looking model that is wrong.
Note the ablation site `o_proj` therefore consumes the **gated** output (input dim 6144 ✓).

### 🔴 R3 · The GDN q/k scaling asymmetry
`q = inv² · rms_norm(q)`, `k = inv · rms_norm(k)`, `inv = 128^-0.5`, **weight=None**.
Squared on q, not on k. Easy to "fix" into symmetry and silently change the model.

### 🟠 R4 · fp32 interiors that look like fp16
`RMSNormGated` computes `silu(z)` and the norm in **fp32** then casts back.
`g = exp(−exp(A_log_f32) · softplus(a + dt_bias))` is fp32. The GDN state is fp32.
The ablation accumulates in fp32. Running any of these in fp16 is a slow-motion accuracy leak that
compounds across 129 sites and 64 layers.

### 🟠 R5 · The inverse Hadamard on the embedding
`token_embd` is in the manifest's **inverse** set, alone. Embedding output is un-rotated
(`fwht(..., inverse=True)` → signs applied *after* the transform, not before). Opposite order.

### 🟠 R6 · Trit repacking correctness
Base-3 decode is `((packed · 3^i) & 255) · 3 >> 8` in three stages (16 B, 8 B, 2 B tail).
*Mitigate:* round-trip the converter against `codec.py`'s independent `unpack()` on every tensor,
assert bit-exact, and fail the build otherwise.

### 🟡 R7 · mrope interleaved + partial rotary
64 of 256 dims rotated, sections [11,11,10], `mrope_interleaved: true`. For text-only all three
position streams are equal so it degenerates to RoPE — **but the interleaved dim layout still
matters**. Validate against Tier 2.

### 🟡 R8 · fp16 activation overflow
Pack activations are fp16. FWHT's 1/√1024 scaling helps, but 27B-scale residuals plus a 17408-wide
FFN can reach fp16 range. Consider bf16 activations throughout — the weights are ternary, so
activation dtype is nearly free.

### 🟡 R9 · CUDA graphs vs dynamic shapes
Capture the **decode** step only (fixed shapes, ~600 launches → one replay). Prefill shapes vary;
leave it eager or bucket it.

### 🔴 R10 · KV traffic (see §0) — the project's defining risk
64 KiB/token → 2 GiB @32K, 16 GiB @262K. The advertised 262K is **KV-bound, not weight-bound**.
Offloading it to host RAM caps decode at **0.8–12 tok/s** depending on context and PCIe gen.
*Mitigate:* 4-bit KV + full residency (§0), ceiling ~84K. **Not optional.**

### 🟠 R11 · 4-bit KV quality drift
KV quantisation is usually benign, but it has never been measured *on this model*, which is
unusual twice over: only 16 layers carry KV, and `head_dim` is 256 (wider groups per head than
typical). Drift will show up as degraded long-range recall, not as garbage — invisible without a
targeted test. *Mitigate:* measure perplexity and a needle-in-haystack retrieval at 8K/32K/64K
against fp16 KV **before** committing; keep a fp16 sink window if needed; make precision a flag.

### 🟠 R12 · sm_120 toolchain freshness
Blackwell consumer needs **CUDA 12.8+** and PyTorch **2.7+/cu128**. `flash-attn` wheels for sm_120
may not exist; `bitsandbytes`-style prebuilt deps often lag too. A stale toolchain fails as
`no kernel image is available for execution on the device` — at runtime, not at build time.
*Mitigate:* pin and assert `torch.cuda.get_device_capability() == (12, 0)` plus a CUDA-version
check at import; compile with `-arch=sm_120`; verify the SDPA fallback path works on day one.

### 🟡 R13 · Laptop thermals and TGP
45–100 W TGP. Sustained decode will clock down where a 30-second benchmark will not.
*Mitigate:* all benchmarks are ≥5-minute sustained runs reporting p50/p95 inter-token latency, not
burst tok/s. Log `nvidia-smi` clocks and power alongside. Agentic workloads are long-running by
nature, so the sustained number is the only honest one.

### 🟡 R14 · Conversion needs host RAM
The source pack is 8.6 GB (or 5.95 GB GGUF) and 32 GB of RAM is comfortable but not infinite if
anything else is running. *Mitigate:* the converter streams tensor-by-tensor and never holds the
whole model; it must be safe to run on the target machine itself.

### ⚪ R15 · Licensing
Pack and runtime are Apache-2.0. The converter may read the format; prefer re-deriving from
`config.json` over copying code, and carry `NOTICE` either way.

### ⚪ R16 · Responsible use
Unchanged by the port. The README's section stays as-is; faster inference is not a reason to drop it.

---

## 8. Milestones

Assumes one engineer. ⛔ = needs NVIDIA hardware (this sandbox has none: no GPU, no nvcc, no torch).

⛔ = needs the target machine. This sandbox has no GPU, no nvcc and no torch, so M0 is the only
milestone that can be built here — which is fine, because M0 is the right first move regardless.

| # | Milestone | Deliverable | Exit criterion | Est. |
|---|---|---|---|---|
| **M0** 🟡 | Converter + Tier-1/2 oracles + llama.cpp baseline | `tools/convert.py`, `bonsai/{codec,config,format}.py` | ✅ converter round-trips **byte-exact**, cross-checked against the authors' own `transcode`; ✅ Tier-1 oracle working; ✅ 56 CPU tests green; ⬜ streaming torch reference; ⬜ **llama.cpp baseline on the target machine** | 1–1.5 wk |
| **M1** ⛔ | K1 GEMV + K3 FWHT | one packed linear on GPU | matches Tier-1 oracle to fp16 tolerance; sm_120 toolchain proven (R12) | 1 wk |
| **M2** ⛔ | K4, K6, K8, **K9 4-bit KV** + decode path | batch 1, greedy, resident KV | logits match Tier 2; top-1 ≥99% vs Tier 3; **32K context resident in <7.6 GiB**; KV drift measured (R11) | 2.5–3 wk |
| **M3** ⛔ | K7 ablation + α / layer controls | `--alpha`, `--layers` | `selfcheck` residual ~1e-6 at **129** sites; α=0 reproduces base bit-exactly | 3 d |
| **M4** ⛔ | K2 GEMM + K5 chunked GDN + **prefix cache** | prefill | pp512 ≥ 400 tok/s sustained; **cached 16K prefix → TTFT < 1 s** | 2–3 wk |
| **M5** ⛔ | CUDA graphs, occupancy, host-paged cold blocks | perf pass | ≥55% of roofline **sustained 5 min** (R13); 64K context usable | 1–1.5 wk |
| **M6** ⛔ | Evals, packaging, docs | CUDA image, README rewrite | README eval table reproduced; agentic acceptance run (§10) green | 1 wk |

**Realistic total: 9–12 weeks.** M4 is the one that slips — chunked GDN prefill is the hardest
kernel in the project. M2 grew because 4-bit KV joined the critical path.

**Sequencing note:** M0's llama.cpp baseline is the highest-value early task. It (a) gives the
end-to-end oracle the whole project depends on, (b) measures real bandwidth, PCIe gen and thermal
behaviour instead of assuming them, and (c) tells us what "fluent" costs today — including whether
`--lora` even applies correctly on CUDA, which has never been verified. A day of benchmarking here
de-risks the following two months.

---

## 9. Proposed layout

```
bonsai/
  __init__.py
  config.py        parse pack config.json → dataclass
  convert.py       → tools/convert_pack.py backend
  reference.py     Tier-2 pure-torch model (slow, obviously correct)
  model.py         native model: layers, cache, forward
  ablation.py      direction loading + the 129-site policy (α, layers)
  kernels/
    __init__.py    load/JIT the extension
    ternary_gemv.cu ternary_gemm.cu fwht.cu gdn_step.cu gdn_chunk.cu
    conv1d.cu residual_ablate.cu
    bindings.cpp
tools/
  convert_pack.py  MLX pack | GGUF → .bonsai, with --check
  bench.py parity.py
tests/
  test_convert.py test_kernels.py test_reference_parity.py test_ablation.py
docker/Dockerfile.cuda
docs/NATIVE-CUDA.md FORMAT.md
```

**Kept as-is:** `directions/`, `gguf/` (demoted to Tier-3 oracle), `swift/`, `LICENSE`,
`scripts/export_gguf_lora.py` (still the provenance path).
**Retired after M6:** `bonsai_abliterate/` (MLX), `run.py`'s MLX path, `docker/Dockerfile` (CPU).
Retire — do not delete — until M2 parity passes; the MLX path is a useful oracle until then.

---

## 10. Acceptance gates

### Correctness

| Gate | Threshold |
|---|---|
| Converter | bit-exact round-trip vs `codec.py::unpack` on all 402 modules |
| K1/K2/K3 | ≤ 2e-3 relative vs Tier-1 fp16 oracle |
| K4/K5 | ≤ fp32 tolerance vs Tier-2, T ∈ {1,2,17,512,4096}, ±mask |
| End-to-end | top-1 agreement ≥ 99% vs llama.cpp fork, fixed 200-prompt set |
| Ablation | **129** sites; residual along `r` ≤ 1e-4, target ~1e-6; α=0 bit-reproduces base |
| 4-bit KV | perplexity within 1% of fp16 KV; needle-retrieval parity at 8K/32K/64K |

### "Fluent, with enough speed for agentic workflows"

Your stated goal, made measurable. These are the numbers that decide whether this worked.

| Gate | Threshold | Why |
|---|---|---|
| Resident footprint | 32K context in **≤ 7.6 GiB**, zero KV offload | removes the §0 bottleneck entirely |
| Decode, sustained | **≥ 25 tok/s** p50 at 32K over a 5-minute run | faster than reading; survives thermals (R13) |
| Inter-token p95 | ≤ 2× p50 | "fluent" means no stalls, not just a good average |
| TTFT, cached prefix | **< 1 s** for a reused 16K prefix | the agentic turn-loop case |
| TTFT, cold | ≤ 45 s for 32K (≥ 400 tok/s prefill, sustained) | tolerable once per session, not per turn |
| Context | 64K usable, 84K ceiling | honest; 262K is not reachable on 8 GB |
| Stability | 1-hour agentic loop, no OOM, no thermal collapse, no drift | "without any problem" |

Stretch: ≥ 30 tok/s at 32K, which the §0 roofline says is achievable at 55% efficiency.

---

## 11. Appendix — retained findings from the MLX investigation

Still load-bearing, because Track-3 oracle work depends on them:

- `mlx[cuda12]==0.32.0` **does** resolve; the dead `mlx-cuda` name (stalled at 0.30.0) is what made
  the README conclude CUDA was impossible. The MLX path can therefore still be stood up as an
  oracle if useful.
- MLX CUDA supports 2-bit affine g128 via `qmv`/`qmm_naive`, but **never** via tensor cores —
  `supports_qmm_sm80`/`sm90` are hardcoded to `bits == 4 || bits == 8`. This is a real part of why
  the native path is justified.
- `mlx-lm`'s gated-delta kernel is Metal-only (`if not mx.metal.is_available(): return None`) and
  falls back to a Python per-timestep loop on CUDA. Our K4/K5 are precisely the kernels MLX lacks.
- The shipped `gguf/bonsai-abliterate-lora.gguf` is verified sound: 258 tensors = 129 pairs, and
  `lora_b == −r` exactly (max deviation 0.000e+00). It is a trustworthy Tier-3 oracle — **but** it
  has never been proven to apply correctly on a CUDA backend, and llama.cpp silently ignores a LoRA
  on a tensor that doesn't route through `build_lora_mm`. Prove it before trusting it (scale 0
  reproduces base, scale 100 destroys the model).

---

## 12. Open items

1. **Measure, don't assume** (M0, one afternoon on the target machine):
   `nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv`
   plus a bandwidth probe. Resolves the 384 vs 448 GB/s conflict, the PCIe gen, and how much VRAM
   the display is already eating — all three feed directly into §0's budget.
2. **Context target.** Plan optimises for **32K resident, 64K usable**. If your agentic workflows
   genuinely need more, say so now — it changes the KV design (host-paged cold blocks, possibly
   sliding-window attention) rather than being bolted on later.
3. **bf16 vs fp16 activations** (R8) — I lean bf16; costs nothing with ternary weights, and
   Blackwell handles it natively.
4. **Vision tower** — plan is text-only, which also saves 0.92 GB. Say if image input must survive;
   on 8 GB it would have to be loaded on demand.
5. **Serving shape** — plan assumes batch 1, single user, streaming. If the agent runs parallel
   tool-calls needing concurrent generations, batching changes M5 materially.
6. **GDN state precision — a free 72 MiB.** The README's measured budget lists the recurrent state
   at **72 MiB**, which is fp16; I budgeted **144 MiB** because the pack config says
   `mamba_ssm_dtype: float32`. Resolve in M2. If fp16 state holds up numerically it buys ~4,600
   extra resident KV tokens for nothing — but R4 warns that this model keeps fp32 interiors
   deliberately, so measure before taking it.
7. **Opportunity, not a commitment:** Blackwell has native **FP4 tensor cores**, and ternary
   {−1,0,+1} is a strict subset of FP4's representable values. Decoding trits → FP4 in shared
   memory (half the smem of fp16) could roughly double prefill throughput with *zero* extra
   quantisation error. Worth a spike during M4; not on the critical path.
