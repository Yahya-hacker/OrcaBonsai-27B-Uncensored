"""Torch primitives: ternary dequantisation, the packed Linear, FWHT, norms.

Every op has two paths. The **kernel path** calls the compiled CUDA extension when it
is available; the **portable path** is plain torch and runs anywhere, including CPU.
They are tested against each other, so the portable path is both a fallback and the
GPU oracle.

That split is deliberate. It means the model is runnable and verifiable *before* the
kernels compile: slower, but producing the same tokens. Compiling then becomes a
performance change with a correctness test attached, rather than the step that decides
whether anything works at all.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import PTQ1_0_STAGES
from .format import GROUPS_PER_SUPERBLOCK, TRIT_BYTES_PER_GROUP

GROUP = 128
SUPERBLOCK = 1024


# ------------------------------------------------------------------ dequantisation

def dequantize_ternary(codes: torch.Tensor, scales: torch.Tensor,
                       out_features: int, in_features: int,
                       dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """``[out, in]`` dense weights from packed trits.

    ``codes``  ``[out, (in/1024)*208]`` uint8, groups of 128 weights in 26 bytes
    ``scales`` ``[out, in/128]`` float16

    Base-3: ``trit_t = ((b * 3**t) & 255) * 3 >> 8``, laid out trit-major within each
    stage, so element ``t*nbytes + i`` comes from byte ``i``'s trit ``t``. Levels are
    ``(code - 1) * scale``, i.e. the ``bias == -scale`` identity baked in.
    """
    n_groups = in_features // GROUP
    if codes.numel() != out_features * n_groups * TRIT_BYTES_PER_GROUP:
        raise ValueError(
            f"codes has {codes.numel()} bytes, expected "
            f"{out_features * n_groups * TRIT_BYTES_PER_GROUP} for "
            f"[{out_features}, {in_features}]")

    g = codes.view(out_features, n_groups, TRIT_BYTES_PER_GROUP).to(torch.int32)
    trits = torch.empty(out_features, n_groups, GROUP,
                        dtype=torch.int32, device=codes.device)
    off = 0
    for lo, hi, count in PTQ1_0_STAGES:
        nb = hi - lo
        b = g[:, :, lo:hi]
        for t in range(count):
            # the kernel uses an incremental carry p = (p*3) & 255; identical result
            trits[:, :, off + t * nb: off + (t + 1) * nb] = \
                ((b * (3 ** t)) & 255) * 3 >> 8
        off += count * nb

    w = (trits - 1).to(dtype).view(out_features, n_groups, GROUP)
    return (w * scales.to(dtype).view(out_features, n_groups, 1)).view(
        out_features, in_features)


# -------------------------------------------------------------------------- FWHT

def fwht(x: torch.Tensor, block: int = SUPERBLOCK,
         signs: torch.Tensor | None = None, inverse: bool = False) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard transform over the last axis, in blocks.

    Forward applies ``signs`` before the transform; inverse applies them after (the
    embedding's convention, R5). ``1/sqrt(block)`` makes it self-inverse and keeps fp16
    activations in range.
    """
    *lead, dim = x.shape
    if dim % block:
        raise ValueError(f"last dim {dim} is not a multiple of block {block}")
    y = x.reshape(-1, dim // block, block).float()
    if signs is not None and not inverse:
        y = y * signs.reshape(1, -1, block).float()
    h = 1
    while h < block:
        y = y.view(-1, dim // block, block // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2)
        h *= 2
    y = y.reshape(-1, dim // block, block) / math.sqrt(block)
    if signs is not None and inverse:
        y = y * signs.reshape(1, -1, block).float()
    return y.reshape(*lead, dim).to(x.dtype)


# -------------------------------------------------------------------------- norms

def rms_norm(x: torch.Tensor, weight: torch.Tensor | None,
             eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    return out.to(x.dtype)


def rms_norm_gated(h: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """``silu(gate) * rms_norm(h)`` with an fp32 interior (R4). Not a plain multiply."""
    normed = h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + eps)
    return (F.silu(gate.float()) * normed * weight.float()).to(h.dtype)


# ------------------------------------------------------------------ packed Linear

class TernaryLinear(nn.Module):
    """A packed ternary projection, bias-free.

    Holds 1.75 bpw codes and per-group scales. ``cache_weight=True`` keeps the
    dequantised matrix, which is fine for small models and tests but is exactly what
    an 8 GB card cannot afford for the real one -- 26.9 B params at fp16 is 54 GB.
    """

    def __init__(self, out_features: int, in_features: int,
                 codes: torch.Tensor, scales: torch.Tensor,
                 hadamard: bool = False, cache_weight: bool = False):
        super().__init__()
        if in_features % SUPERBLOCK:
            raise ValueError(f"in_features {in_features} is not a multiple of 1024")
        self.out_features, self.in_features = out_features, in_features
        self.hadamard = hadamard
        self.cache_weight = cache_weight
        self.register_buffer("codes", codes, persistent=True)
        self.register_buffer("scales", scales, persistent=True)
        self.register_buffer("signs", None, persistent=False)
        self._cached: torch.Tensor | None = None

    def extra_repr(self) -> str:
        return (f"{self.in_features} -> {self.out_features}, "
                f"hadamard={self.hadamard}, bpw=1.75")

    def weight(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        if self._cached is not None:
            return self._cached
        w = dequantize_ternary(self.codes, self.scales,
                               self.out_features, self.in_features, dtype)
        if self.cache_weight:
            self._cached = w
        return w

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """The input-side Hadamard. Hoisted out of ``forward`` so a block can run it
        once for several projections that share an input (q/k/v, gate/up, in_proj_*)."""
        if not self.hadamard:
            return x
        return fwht(x, SUPERBLOCK, self.signs)

    def forward(self, x: torch.Tensor, transformed: bool = False) -> torch.Tensor:
        if self.hadamard and not transformed:
            x = self.transform(x)
        return F.linear(x, self.weight(x.dtype))


class DenseLinear(nn.Module):
    """An unpacked projection (``in_proj_a``/``in_proj_b``), kept in fp32."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("weight_", weight, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float(), self.weight_.float())


# --------------------------------------------------------------------- ablation op

def residual_ablate(x: torch.Tensor, y: torch.Tensor, direction: torch.Tensor,
                    alpha: float | torch.Tensor) -> torch.Tensor:
    """``x + y - alpha * dot(y, r) * r``, fp32 interior.

    When alpha is exactly 0 this must be bit-identical to ``x + y``, which the test
    suite asserts -- so the correction is skipped rather than multiplied by zero.
    """
    if isinstance(alpha, (int, float)) and alpha == 0.0:
        return (x.float() + y.float()).to(y.dtype)
    yf = y.float()
    comp = (yf * direction).sum(-1, keepdim=True)
    return (x.float() + yf - alpha * comp * direction).to(y.dtype)


def ablate(y: torch.Tensor, direction: torch.Tensor,
           alpha: float | torch.Tensor) -> torch.Tensor:
    if isinstance(alpha, (int, float)) and alpha == 0.0:
        return y
    yf = y.float()
    comp = (yf * direction).sum(-1, keepdim=True)
    return (yf - alpha * comp * direction).to(y.dtype)


# --------------------------------------------------------------------------- RoPE

def build_rope(rotary_dim: int, max_pos: int, theta: float,
               device=None) -> tuple[torch.Tensor, torch.Tensor]:
    half = rotary_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32,
                                        device=device) * 2.0 / rotary_dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    ang = torch.outer(t, inv)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               rotary_dim: int) -> torch.Tensor:
    """Partial rotary over ``[..., seq, head_dim]``; only the first ``rotary_dim`` turn."""
    half = rotary_dim // 2
    a = x[..., :half].float()
    b = x[..., half:rotary_dim].float()
    c = cos.view(*([1] * (x.dim() - 2)), -1, half)
    s = sin.view(*([1] * (x.dim() - 2)), -1, half)
    return torch.cat([(a * c - b * s).to(x.dtype),
                      (a * s + b * c).to(x.dtype),
                      x[..., rotary_dim:]], dim=-1)
