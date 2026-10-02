"""Quantised, paged KV cache -- the component the project's fluency actually depends on.

The problem it solves
---------------------
Only 16 of 64 layers carry a KV cache, but at fp16 those still cost **64 KiB per
token**: 2 GiB at 32K context, 16 GiB at the full 262K. On an 8 GB card the weights
already take 5.53 GiB, so an fp16 cache has to spill to system RAM -- and decode
re-reads the *entire* cache every token, so over PCIe that caps throughput at roughly
6 tok/s at 32K and under 2 tok/s at 262K. Measured against ~15 ms to read all 26.9 B
weights from VRAM, KV transfer is 6-50x more expensive than the rest of the model
combined.

So the cache must stay resident, which means it must shrink.

The design
----------
* **Group-wise affine int4** along ``head_dim``, with **a finer group for K than V**.
  K errors perturb attention *logits*, which are then exponentiated, so they compound;
  V errors enter the output linearly and partly average out. K therefore gets group 32
  (5.00 bpw) and V group 64 (4.50 bpw), averaging 4.75 bpw -- 19 KiB/token, a 3.4x
  reduction, ~74K tokens resident in a 1.34 GiB budget. Paying 5% more memory than a
  uniform group-64 cache for materially better logits is the right trade.
* **Asymmetric by default.** K in particular is poorly centred; a zero point costs 16
  bits per group and buys real accuracy. ``symmetric=True`` saves 0.25-0.5 bpw.

Measured error, and why ~10% is expected
----------------------------------------
int4 holds 16 levels across each group's min-max span. For a group of 64 roughly
Gaussian values that span is about 5.8 sigma, so the step is 0.385 sigma and the RMS
quantisation error is ``step/sqrt(12) = 0.111 sigma`` -- about **10% relative error per
element, by construction**. Measured values track that theory closely:

===========  ========  ==========  =========================
group/mode   bits/val  rel. error  rel. error, outlier-heavy
===========  ========  ==========  =========================
32 asym          5.00      0.078                       0.101
64 asym          4.50      0.090                       0.132
128 asym         4.25      0.101                       0.184
===========  ========  ==========  =========================

Two things follow. Group 128 is a trap: it saves 0.25 bits and loses 40% more accuracy
once outlier channels are present, which attention K/V genuinely have. And a 10%
elementwise error is survivable only because softmax is contractive and the sink window
protects recent tokens -- it is *not* self-evidently fine. R11 stands: measure
perplexity and needle-retrieval on the real model before trusting this.
* **Paged blocks** of 128 tokens, so growth is amortised, cold pages can later be moved
  to host RAM without touching the hot path, and an agentic prefix can be shared.
* **An fp16 sink window.** The most recent ``sink`` tokens stay unquantised. Recent
  context is what decode attends to most sharply, and this is the standard, cheap
  insurance against quantisation drift.

Nothing here is believed to be free: :func:`measure_error` exists so the accuracy cost
is measured on the real model rather than assumed (R11).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT, BonsaiConfig

F32 = np.float32
PAGE_TOKENS = 128


# --------------------------------------------------------------------- quantisation

def quantize(x: np.ndarray, group: int = 64, symmetric: bool = False):
    """Affine int4 along the last axis. Returns ``(codes uint8, scale, zero)``.

    Two 4-bit codes are packed per byte, low nibble first.
    """
    *lead, dim = x.shape
    if dim % group:
        raise ValueError(f"last dim {dim} is not a multiple of group {group}")
    xf = x.astype(F32).reshape(-1, dim // group, group)

    if symmetric:
        amax = np.abs(xf).max(axis=-1, keepdims=True)
        scale = np.maximum(amax, 1e-8) / 7.0
        zero = np.zeros_like(scale)
        q = np.clip(np.rint(xf / scale) + 8, 0, 15)
    else:
        lo = xf.min(axis=-1, keepdims=True)
        hi = xf.max(axis=-1, keepdims=True)
        scale = np.maximum(hi - lo, 1e-8) / 15.0
        zero = lo
        q = np.clip(np.rint((xf - zero) / scale), 0, 15)

    q = q.astype(np.uint8).reshape(-1, dim // group, group // 2, 2)
    codes = (q[..., 0] | (q[..., 1] << 4)).astype(np.uint8)
    shape = (*lead, dim // group, group // 2)
    return (codes.reshape(shape),
            scale.astype(np.float16).reshape(*lead, dim // group),
            zero.astype(np.float16).reshape(*lead, dim // group))


def dequantize(codes: np.ndarray, scale: np.ndarray, zero: np.ndarray,
               group: int = 64, symmetric: bool = False) -> np.ndarray:
    *lead, n_groups, half = codes.shape
    lo = (codes & 0x0F).astype(F32)
    hi = (codes >> 4).astype(F32)
    q = np.stack([lo, hi], axis=-1).reshape(*lead, n_groups, group)
    s = scale.astype(F32)[..., None]
    z = zero.astype(F32)[..., None]
    out = (q - 8.0) * s if symmetric else q * s + z
    return out.reshape(*lead, n_groups * group)


def bits_per_value(group: int = 64, symmetric: bool = False) -> float:
    return (group * 4 + (16 if symmetric else 32)) / group


# ------------------------------------------------------------------------- the cache

@dataclass
class KVConfig:
    k_group: int = 32          #: finer: K errors feed the softmax and compound
    v_group: int = 64          #: coarser: V errors enter linearly
    symmetric: bool = False
    sink: int = 128            #: most recent tokens kept in fp16
    page: int = PAGE_TOKENS

    @property
    def group(self) -> int:    # back-compat for callers that want a single number
        return self.v_group

    def bits_per_token_value(self) -> float:
        return 0.5 * (bits_per_value(self.k_group, self.symmetric)
                      + bits_per_value(self.v_group, self.symmetric))

    def bytes_per_token(self, cfg: BonsaiConfig = DEFAULT) -> int:
        """Across all 16 full-attention layers, K and V."""
        n_full = cfg.num_layers // cfg.full_attention_interval
        per_plane = n_full * cfg.num_key_value_heads * cfg.head_dim
        return int(per_plane * (bits_per_value(self.k_group, self.symmetric)
                                + bits_per_value(self.v_group, self.symmetric)) / 8)


class LayerKVCache:
    """One full-attention layer's cache: quantised pages plus an fp16 sink window."""

    def __init__(self, num_heads: int, head_dim: int, kvcfg: KVConfig | None = None):
        self.h, self.d = num_heads, head_dim
        self.cfg = kvcfg or KVConfig()
        self._pages: list[tuple] = []          # (k_codes, k_s, k_z, v_codes, v_s, v_z)
        self._sink_k: list[np.ndarray] = []    # fp16, most recent
        self._sink_v: list[np.ndarray] = []
        self.length = 0

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        """``k``/``v``: ``[tokens, heads, head_dim]``."""
        if k.shape != v.shape or k.shape[1:] != (self.h, self.d):
            raise ValueError(f"expected [tokens, {self.h}, {self.d}], got {k.shape}")
        for t in range(k.shape[0]):
            self._sink_k.append(k[t].astype(np.float16))
            self._sink_v.append(v[t].astype(np.float16))
            self.length += 1
        self._spill()

    def _spill(self) -> None:
        """Move everything older than the sink window into quantised pages."""
        while len(self._sink_k) > self.cfg.sink + self.cfg.page:
            ktile = np.stack(self._sink_k[:self.cfg.page])
            vtile = np.stack(self._sink_v[:self.cfg.page])
            del self._sink_k[:self.cfg.page], self._sink_v[:self.cfg.page]
            self._pages.append((
                *quantize(ktile, self.cfg.k_group, self.cfg.symmetric),
                *quantize(vtile, self.cfg.v_group, self.cfg.symmetric)))

    def gather(self):
        """Reconstruct the full ``(K, V)`` as ``[heads, length, head_dim]`` fp32."""
        ks, vs = [], []
        for kc, ksc, kz, vc, vsc, vz in self._pages:
            ks.append(dequantize(kc, ksc, kz, self.cfg.k_group, self.cfg.symmetric))
            vs.append(dequantize(vc, vsc, vz, self.cfg.v_group, self.cfg.symmetric))
        if self._sink_k:
            ks.append(np.stack(self._sink_k).astype(F32))
            vs.append(np.stack(self._sink_v).astype(F32))
        K = np.concatenate(ks, axis=0) if ks else np.zeros((0, self.h, self.d), F32)
        V = np.concatenate(vs, axis=0) if vs else np.zeros((0, self.h, self.d), F32)
        return K.transpose(1, 0, 2), V.transpose(1, 0, 2)

    def nbytes(self) -> int:
        """Actual bytes held, quantised pages plus the unquantised sink window."""
        q = sum(a.nbytes for page in self._pages for a in page)
        return q + sum(a.nbytes for a in self._sink_k + self._sink_v)

    def quantized_tokens(self) -> int:
        return self.length - len(self._sink_k)

    def flush(self) -> None:
        """Quantise everything, including the sink. Used when a prefix goes cold."""
        while self._sink_k:
            n = min(self.cfg.page, len(self._sink_k))
            ktile = np.stack(self._sink_k[:n]); vtile = np.stack(self._sink_v[:n])
            del self._sink_k[:n], self._sink_v[:n]
            self._pages.append((
                *quantize(ktile, self.cfg.k_group, self.cfg.symmetric),
                *quantize(vtile, self.cfg.v_group, self.cfg.symmetric)))


class KVCache:
    """All 16 full-attention layers. Linear-attention layers hold recurrent state
    instead, which is context-independent -- see ``bonsai.state``."""

    def __init__(self, cfg: BonsaiConfig = DEFAULT, kvcfg: KVConfig | None = None):
        self.cfg, self.kvcfg = cfg, kvcfg or KVConfig()
        self.layers = {
            i: LayerKVCache(cfg.num_key_value_heads, cfg.head_dim, self.kvcfg)
            for i in range(cfg.num_layers) if cfg.is_full_attention(i)
        }

    def __getitem__(self, layer: int) -> LayerKVCache:
        if layer not in self.layers:
            raise KeyError(f"layer {layer} is linear-attention; it has no KV cache")
        return self.layers[layer]

    @property
    def length(self) -> int:
        return next(iter(self.layers.values())).length if self.layers else 0

    def nbytes(self) -> int:
        return sum(l.nbytes() for l in self.layers.values())

    def plan(self, vram_gib: float = 8.0, **kw) -> dict:
        """Max resident context under this KV configuration."""
        per_tok = self.kvcfg.bytes_per_token(self.cfg)
        p = self.cfg.memory_plan(vram_gib=vram_gib, **kw)
        p["kv_bytes_per_token"] = per_tok
        p["max_resident_context"] = int(max(p["kv_free_bytes"], 0) // per_tok)
        return p


# -------------------------------------------------------------------- measurement

def measure_error(x: np.ndarray, group: int = 64, symmetric: bool = False) -> dict:   # noqa: D401
    """Round-trip error for a real activation tensor. Run this before trusting int4."""
    q = dequantize(*quantize(x, group, symmetric), group, symmetric)
    err = q - x.astype(F32)
    denom = np.linalg.norm(x.astype(F32))
    return {
        "rel_fro": float(np.linalg.norm(err) / max(denom, 1e-12)),
        "max_abs": float(np.abs(err).max()),
        "bits_per_value": bits_per_value(group, symmetric),
        "cosine": float(
            (x.astype(F32).ravel() @ q.ravel())
            / max(denom * np.linalg.norm(q), 1e-12)),
    }
