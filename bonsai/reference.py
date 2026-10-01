"""Tier-2 oracle: obviously-correct reference math for every Bonsai 2 layer.

Slow, readable, numpy. This is the regression net the CUDA kernels are graded against,
and it is deliberately written for *inspectability* over speed -- when a kernel and
this disagree, the kernel is wrong until proven otherwise.

Why numpy and not torch
-----------------------
A materialised fp16 reference model is 54 GB, which fits neither 8 GB of VRAM nor
32 GB of RAM (see PLAN-NATIVE-CUDA.md §6). The usable reference is therefore
*streaming*: dequantise one matrix at a time from the container, use it, drop it. That
makes the arithmetic the interesting part, and the arithmetic is backend-agnostic.
Every function here takes plain arrays, so the same code validates a torch path.

The four things this file exists to get right
---------------------------------------------
Each of these is a silent-failure mode -- wrong output, no error:

* **R2** ``q_proj`` emits ``24 * 256 * 2`` and the *second half is a gate*, applied as
  ``o_proj(attn * sigmoid(gate))``. Miss it and the model is plausible and wrong.
* **R3** the GDN scales q by ``inv**2`` and k by ``inv`` where ``inv = 128**-0.5``.
  Squared on one side only. "Fixing" the asymmetry silently changes the model.
* **R4** ``RMSNormGated`` computes silu and the norm in **fp32** then casts back; the
  GDN decay and state are fp32. Running them at activation precision leaks accuracy
  across 64 layers.
* **R5** the embedding carries the *inverse* Hadamard -- signs after the transform,
  not before.
"""
from __future__ import annotations

import numpy as np

from .config import DEFAULT, BonsaiConfig

F32 = np.float32


# --------------------------------------------------------------------- elementwise

def silu(x: np.ndarray) -> np.ndarray:
    x = x.astype(F32)
    return x / (1.0 + np.exp(-x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(F32)))).astype(F32)


def softplus(x: np.ndarray) -> np.ndarray:
    x = x.astype(F32)
    return np.logaddexp(x, 0.0).astype(F32)


def rms_norm(x: np.ndarray, weight: np.ndarray | None, eps: float = 1e-6) -> np.ndarray:
    """Always reduces in fp32. ``weight=None`` matches mx.fast.rms_norm's unweighted form,
    which the GDN uses on q and k."""
    xf = x.astype(F32)
    scale = 1.0 / np.sqrt((xf * xf).mean(axis=-1, keepdims=True) + eps)
    out = xf * scale
    return out if weight is None else out * weight.astype(F32)


def rms_norm_gated(h: np.ndarray, gate: np.ndarray, weight: np.ndarray,
                   eps: float = 1e-6) -> np.ndarray:
    """mlx_lm's ``_precise_swiglu``: ``silu(gate_f32) * rms_norm(h)_f32``, then cast.

    Not a plain multiply, and the fp32 interior is load-bearing (R4).
    """
    normed = rms_norm(h, weight, eps)
    return (silu(gate) * normed).astype(h.dtype)


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    return (silu(gate) * up.astype(F32)).astype(up.dtype)


# ----------------------------------------------------------------------- hadamard

def fwht(x: np.ndarray, block: int = 1024, signs: np.ndarray | None = None,
         inverse: bool = False) -> np.ndarray:
    """Fast Walsh-Hadamard transform over the last axis, in blocks.

    Forward order is ``signs`` then transform; the embedding uses the inverse, which is
    transform then ``signs`` (R5). Normalised by ``1/sqrt(block)`` so it is orthogonal
    and self-inverse -- that 1/32 at block 1024 is also what keeps fp16 activations in
    range (R8).
    """
    shape = x.shape
    if shape[-1] % block:
        raise ValueError(f"last dim {shape[-1]} is not a multiple of block {block}")
    y = x.astype(F32).reshape(-1, shape[-1] // block, block)
    if signs is not None and not inverse:
        y = y * signs.astype(F32).reshape(1, -1, block)
    h = 1
    while h < block:
        y = y.reshape(-1, shape[-1] // block, block // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = np.stack([a + b, a - b], axis=-2)
        h *= 2
    y = y.reshape(-1, shape[-1] // block, block) / np.sqrt(block, dtype=F32)
    if signs is not None and inverse:
        y = y * signs.astype(F32).reshape(1, -1, block)
    return y.reshape(shape)


# ---------------------------------------------------------------------- ablation

def ablate(y: np.ndarray, direction: np.ndarray, alpha: float) -> np.ndarray:
    """``y - alpha * dot(y, r) * r``, fp32 interior. Mirrors bonsai_abliterate."""
    if alpha == 0.0:
        return y
    yf = y.astype(F32)
    r = direction.astype(F32)
    comp = (yf * r).sum(axis=-1, keepdims=True)
    return (yf - alpha * comp * r).astype(y.dtype)


# --------------------------------------------------------------------- attention

def rope(x: np.ndarray, positions: np.ndarray, rotary_dim: int,
         theta: float = 1e7) -> np.ndarray:
    """Partial rotary: only the first ``rotary_dim`` of ``head_dim`` are rotated.

    At ``partial_rotary_factor 0.25`` and ``head_dim 256`` that is 64 dims; the other
    192 pass through untouched (R7).
    """
    *lead, head_dim = x.shape
    if rotary_dim > head_dim:
        raise ValueError("rotary_dim exceeds head_dim")
    out = x.astype(F32).copy()
    half = rotary_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=F32) * 2.0 / rotary_dim))
    ang = positions.astype(F32).reshape(-1, 1) * inv.reshape(1, -1)
    cos, sin = np.cos(ang), np.sin(ang)
    while cos.ndim < len(lead) + 1:
        cos, sin = cos[None, ...], sin[None, ...]
    # .copy() is load-bearing: these are views into `out`, so without it the write to
    # the first half corrupts `a` before the second half reads it. A rotation must
    # preserve norm, which is what caught this.
    a = out[..., :half].copy()
    b = out[..., half:rotary_dim].copy()
    out[..., :half] = a * cos - b * sin
    out[..., half:rotary_dim] = a * sin + b * cos
    return out


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = True,
              scale: float | None = None) -> np.ndarray:
    """Softmax attention, fp32, with GQA head repetition. ``q`` [H, L, D]."""
    H, L, D = q.shape
    kv_heads = k.shape[0]
    if H % kv_heads:
        raise ValueError(f"{H} query heads is not a multiple of {kv_heads} kv heads")
    rep = H // kv_heads
    k = np.repeat(k, rep, axis=0)
    v = np.repeat(v, rep, axis=0)
    s = (scale if scale is not None else D ** -0.5)
    logits = np.einsum("hqd,hkd->hqk", q.astype(F32), k.astype(F32)) * s
    if causal:
        mask = np.triu(np.ones((L, k.shape[1]), dtype=bool), k=k.shape[1] - L + 1)
        logits = np.where(mask, -np.inf, logits)
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=-1, keepdims=True)
    return np.einsum("hqk,hkd->hqd", p, v.astype(F32))


def apply_output_gate(attn_out: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """R2: ``attn_out * sigmoid(gate)``, applied *before* o_proj.

    ``gate`` is the second half of q_proj's output, reshaped like the attention output.
    """
    if attn_out.shape != gate.shape:
        raise ValueError(f"gate shape {gate.shape} != attention output {attn_out.shape}")
    return attn_out.astype(F32) * sigmoid(gate)


def split_q_and_gate(qproj_out: np.ndarray, num_heads: int, head_dim: int):
    """``[..., num_heads*head_dim*2]`` -> ``(q [..., H, D], gate [..., H*D])``."""
    expected = num_heads * head_dim * 2
    if qproj_out.shape[-1] != expected:
        raise ValueError(f"q_proj width {qproj_out.shape[-1]}, expected {expected}")
    q, gate = np.split(qproj_out, 2, axis=-1)
    return q.reshape(*q.shape[:-1], num_heads, head_dim), gate


# --------------------------------------------------------------- gated delta net

def gdn_qk_scale(q: np.ndarray, k: np.ndarray, head_k_dim: int = 128):
    """R3: ``q <- inv**2 * rms_norm(q)``, ``k <- inv * rms_norm(k)``, weight=None.

    The square is on q only. This asymmetry is real; do not "correct" it.
    """
    inv = F32(head_k_dim ** -0.5)
    return (inv * inv) * rms_norm(q, None, 1e-6), inv * rms_norm(k, None, 1e-6)


def gdn_decay(a: np.ndarray, A_log: np.ndarray, dt_bias: np.ndarray) -> np.ndarray:
    """``g = exp(-exp(A_log) * softplus(a + dt_bias))``, fp32 throughout (R4)."""
    return np.exp(-np.exp(A_log.astype(F32)) * softplus(a + dt_bias)).astype(F32)


def gdn_step(state: np.ndarray, q: np.ndarray, k: np.ndarray, v: np.ndarray,
             g: np.ndarray, beta: np.ndarray):
    """One decode step of the gated delta rule.

    ``state`` [nv, dv, dk] fp32, carried across tokens and context-independent -- which
    is why 48 of 64 layers cost nothing as context grows.

        state = state * g
        kv    = (state . k)
        delta = (v - kv) * beta
        state = state + outer(delta, k)
        y     = (state . q)

    ``q``/``k`` have nk heads and are shared across nv/nk value heads.
    """
    nv, dv, dk = state.shape
    nk = k.shape[0]
    if nv % nk:
        raise ValueError(f"{nv} value heads is not a multiple of {nk} key heads")
    kk = np.repeat(k, nv // nk, axis=0).astype(F32)       # [nv, dk]
    qq = np.repeat(q, nv // nk, axis=0).astype(F32)
    s = state * g.reshape(nv, 1, 1).astype(F32)
    kv = np.einsum("hvd,hd->hv", s, kk)                   # [nv, dv]
    delta = (v.astype(F32) - kv) * beta.reshape(nv, 1).astype(F32)
    s = s + np.einsum("hv,hd->hvd", delta, kk)
    y = np.einsum("hvd,hd->hv", s, qq)
    return s, y


def causal_conv1d(x: np.ndarray, weight: np.ndarray,
                  state: np.ndarray | None = None):
    """Depthwise causal conv over ``[L, C]`` with ``weight [C, K]``.

    ``state`` carries the trailing ``K-1`` inputs between calls, as decode requires.
    """
    L, C = x.shape
    K = weight.shape[-1]
    if state is None:
        state = np.zeros((K - 1, C), dtype=F32)
    padded = np.concatenate([state.astype(F32), x.astype(F32)], axis=0)
    out = np.zeros((L, C), dtype=F32)
    for t in range(L):
        window = padded[t:t + K]                       # [K, C]
        out[t] = (window * weight.astype(F32).T).sum(axis=0)
    return out, padded[-(K - 1):] if K > 1 else state


# ------------------------------------------------------------------------ helpers

def layer_kind(layer: int, cfg: BonsaiConfig = DEFAULT) -> str:
    return "full_attention" if cfg.is_full_attention(layer) else "linear_attention"
