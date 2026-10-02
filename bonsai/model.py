"""The Bonsai 2 27B model in torch, with runtime refusal ablation.

Structure, per the verified architecture (``bonsai/config.py``):

* 64 blocks. ``i % 4 == 3`` is full attention (16 of them), the rest are gated delta
  nets (48). Only the full-attention blocks hold a KV cache, which is why context is
  affordable at all.
* Full attention: 24 query heads, 4 KV heads, head_dim 256, **an output gate**
  (``q_proj`` emits ``24*256*2`` and the second half gates the attention output), and
  partial RoPE over 64 of 256 dims.
* Gated delta net: 48 value heads / 16 key heads, head_dim 128, a depthwise causal
  conv of width 4, and an fp32 recurrent state of ``[48, 128, 128]``.
* SwiGLU MLP, 5120 -> 17408 -> 5120.

**The ablation.** 129 residual writers get ``y -= alpha * (y . r) r`` applied to their
*output*, fused into the residual add. Weights are never modified, so ``alpha`` is a
per-request knob and ``alpha = 0`` is bit-identical to the base model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEFAULT, BonsaiConfig
from .torch_ops import (DenseLinear, TernaryLinear, ablate, apply_rope, build_rope,
                        fwht, residual_ablate, rms_norm, rms_norm_gated)


@dataclass
class RuntimeConfig:
    dtype: torch.dtype = torch.float16
    device: str = "cpu"
    alpha: float = 1.0                 #: refusal ablation strength
    use_kernels: bool = True
    cache_weights: bool = False        #: only for tiny models / tests
    kv_bits: int = 4
    max_position: int = 262144


class GDNState:
    """Per-sequence recurrent state for one gated-delta layer. Context-independent."""

    def __init__(self, cfg: BonsaiConfig, device, dtype=torch.float32):
        nv, dv, dk = (cfg.linear_num_value_heads, cfg.linear_value_head_dim,
                      cfg.linear_key_head_dim)
        self.state = torch.zeros(nv, dv, dk, device=device, dtype=dtype)
        self.conv = torch.zeros(cfg.linear_conv_kernel_dim - 1, cfg.conv_dim,
                                device=device, dtype=torch.float32)

    def reset(self):
        self.state.zero_(); self.conv.zero_()


# --------------------------------------------------------------------------- MLP

class BonsaiMLP(nn.Module):
    def __init__(self, gate: TernaryLinear, up: TernaryLinear, down: TernaryLinear):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = gate, up, down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate_proj and up_proj share an input: transform once, not twice.
        xt = self.gate_proj.transform(x)
        g = self.gate_proj(xt, transformed=True)
        u = self.up_proj(xt, transformed=True)
        h = (F.silu(g.float()) * u.float()).to(x.dtype)
        return self.down_proj(h)


# --------------------------------------------------------------------- attention

class BonsaiAttention(nn.Module):
    """Full attention with the output gate. 16 of 64 layers."""

    def __init__(self, cfg: BonsaiConfig, q, k, v, o, q_norm=None, k_norm=None):
        super().__init__()
        self.cfg = cfg
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = q, k, v, o
        self.q_norm, self.k_norm = q_norm, k_norm
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim

    def forward(self, x, cos, sin, cache=None, positions=None):
        B, L, _ = x.shape
        xt = self.q_proj.transform(x)            # q/k/v share one input
        qg = self.q_proj(xt, transformed=True)
        k = self.k_proj(xt, transformed=True)
        v = self.v_proj(xt, transformed=True)

        # R2: the second half of q_proj is a sigmoid gate on the attention output.
        q, gate = qg.chunk(2, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_kv, self.head_dim)
        v = v.view(B, L, self.n_kv, self.head_dim)

        if self.q_norm is not None:
            q = rms_norm(q, self.q_norm, self.cfg.rms_norm_eps)
        if self.k_norm is not None:
            k = rms_norm(k, self.k_norm, self.cfg.rms_norm_eps)

        q = apply_rope(q.transpose(1, 2), cos, sin, self.cfg.rotary_dim)
        k = apply_rope(k.transpose(1, 2), cos, sin, self.cfg.rotary_dim)
        v = v.transpose(1, 2)

        if cache is not None:
            k, v = cache.update(k, v)

        rep = self.n_heads // self.n_kv
        out = F.scaled_dot_product_attention(
            q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1),
            is_causal=(L > 1))
        out = out.transpose(1, 2).reshape(B, L, self.n_heads * self.head_dim)
        out = (out.float() * torch.sigmoid(gate.float())).to(x.dtype)
        return self.o_proj(out)


class SimpleKV:
    """fp16 KV for correctness runs. The 4-bit paged cache lives in bonsai/kvcache.py
    and is wired in by the engine when ``kv_bits < 16``."""

    def __init__(self):
        self.k = self.v = None

    def update(self, k, v):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
        return self.k, self.v

    @property
    def length(self):
        return 0 if self.k is None else self.k.shape[2]


# --------------------------------------------------------------- gated delta net

class BonsaiGatedDelta(nn.Module):
    """Linear attention. 48 of 64 layers, and the reason long context is affordable."""

    def __init__(self, cfg: BonsaiConfig, qkv, z, a, b, out, conv1d,
                 A_log, dt_bias, norm):
        super().__init__()
        self.cfg = cfg
        self.in_proj_qkv, self.in_proj_z, self.out_proj = qkv, z, out
        self.in_proj_a, self.in_proj_b = a, b
        self.register_buffer("conv1d", conv1d, persistent=True)
        self.register_buffer("A_log", A_log, persistent=True)
        self.register_buffer("dt_bias", dt_bias, persistent=True)
        self.register_buffer("norm", norm, persistent=True)
        self.nk, self.nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.dk, self.dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim

    def forward(self, x, state: GDNState):
        B, L, _ = x.shape
        if B != 1:
            raise NotImplementedError("gated delta net currently runs batch 1")
        xt = self.in_proj_qkv.transform(x)       # qkv, z, a, b share an input
        qkv = self.in_proj_qkv(xt, transformed=True)
        z = self.in_proj_z(xt, transformed=True)
        a = self.in_proj_a(x)                    # unpacked, fp32, no Hadamard
        b = self.in_proj_b(x)

        # depthwise causal conv over the full qkv stream, state carried across calls
        seq = qkv[0].float()
        padded = torch.cat([state.conv, seq], dim=0)
        w = self.conv1d.float().view(self.cfg.conv_dim, self.cfg.linear_conv_kernel_dim)
        K = w.shape[-1]
        conv_out = torch.stack(
            [(padded[t:t + K] * w.T).sum(0) for t in range(L)], dim=0)
        state.conv = padded[-(K - 1):].clone() if K > 1 else state.conv
        conv_out = F.silu(conv_out)

        kd, vd = self.cfg.key_dim, self.cfg.value_dim
        q, k, v = conv_out.split([kd, kd, vd], dim=-1)
        q = q.view(L, self.nk, self.dk)
        k = k.view(L, self.nk, self.dk)
        v = v.view(L, self.nv, self.dv)

        # R3: inv^2 on q, inv on k. Asymmetric on purpose.
        inv = self.dk ** -0.5
        q = (inv * inv) * rms_norm(q.float(), None, 1e-6)
        k = inv * rms_norm(k.float(), None, 1e-6)

        g = torch.exp(-torch.exp(self.A_log.float())
                      * F.softplus(a[0].float() + self.dt_bias.float()))   # [L, nv]
        beta = torch.sigmoid(b[0].float())                                  # [L, nv]

        rep = self.nv // self.nk
        ys = []
        s = state.state
        for t in range(L):
            kt = k[t].repeat_interleave(rep, 0)          # [nv, dk]
            qt = q[t].repeat_interleave(rep, 0)
            s = s * g[t].view(self.nv, 1, 1)
            kv = torch.einsum("hvd,hd->hv", s, kt)
            delta = (v[t].float() - kv) * beta[t].view(self.nv, 1)
            s = s + torch.einsum("hv,hd->hvd", delta, kt)
            ys.append(torch.einsum("hvd,hd->hv", s, qt))
        state.state = s
        y = torch.stack(ys, 0)                           # [L, nv, dv]

        y = rms_norm_gated(y, z.view(L, self.nv, self.dv).float(), self.norm)
        return self.out_proj(y.reshape(1, L, self.nv * self.dv).to(x.dtype))


# -------------------------------------------------------------------------- block

class BonsaiBlock(nn.Module):
    def __init__(self, cfg: BonsaiConfig, layer_idx: int, mixer, mlp,
                 input_ln, post_ln):
        super().__init__()
        self.cfg, self.layer_idx = cfg, layer_idx
        self.mixer, self.mlp = mixer, mlp
        self.register_buffer("input_layernorm", input_ln, persistent=True)
        self.register_buffer("post_attention_layernorm", post_ln, persistent=True)
        self.is_full = cfg.is_full_attention(layer_idx)

    def forward(self, x, *, direction=None, alpha=0.0, **kw):
        h = rms_norm(x, self.input_layernorm, self.cfg.rms_norm_eps)
        if self.is_full:
            y = self.mixer(h, kw["cos"], kw["sin"], kw.get("cache"))
        else:
            y = self.mixer(h, kw["state"])
        # residual writer #1: the mixer's out_proj / o_proj
        x = (residual_ablate(x, y, direction, alpha) if direction is not None
             else (x.float() + y.float()).to(x.dtype))

        h = rms_norm(x, self.post_attention_layernorm, self.cfg.rms_norm_eps)
        y = self.mlp(h)
        # residual writer #2: mlp.down_proj
        return (residual_ablate(x, y, direction, alpha) if direction is not None
                else (x.float() + y.float()).to(x.dtype))


# -------------------------------------------------------------------------- model

class BonsaiForCausalLM(nn.Module):
    def __init__(self, cfg: BonsaiConfig, rt: RuntimeConfig):
        super().__init__()
        self.cfg, self.rt = cfg, rt
        self.blocks = nn.ModuleList()
        self.embed_tokens: TernaryLinear | None = None
        self.lm_head: TernaryLinear | None = None
        self.register_buffer("norm", None, persistent=True)
        self.register_buffer("direction", None, persistent=False)
        self.alpha = rt.alpha
        cos, sin = build_rope(cfg.rotary_dim, min(rt.max_position, 1 << 15),
                              cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # ------------------------------------------------------------------- forward
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Embedding lookup.

        Asymmetric site: a row *is* a residual vector, so the ablation is applied to
        the row itself with no incoming residual. The matrix also carries the inverse
        Hadamard (R5), which is why the lookup is a gather, not a transform.
        """
        w = self.embed_tokens.weight(self.rt.dtype)
        e = F.embedding(ids, w)
        if self.direction is not None and self.alpha != 0.0:
            e = ablate(e, self.direction, self.alpha)
        return e

    def forward(self, ids: torch.Tensor, caches=None, states=None,
                offset: int = 0) -> torch.Tensor:
        B, L = ids.shape
        x = self.embed(ids)
        cos = self.rope_cos[offset:offset + L]
        sin = self.rope_sin[offset:offset + L]
        for i, blk in enumerate(self.blocks):
            x = blk(x, direction=self.direction, alpha=self.alpha,
                    cos=cos, sin=sin,
                    cache=None if caches is None else caches.get(i),
                    state=None if states is None else states.get(i))
        x = rms_norm(x, self.norm, self.cfg.rms_norm_eps)
        return F.linear(x, self.lm_head.weight(x.dtype))

    # --------------------------------------------------------------------- state
    def new_caches(self):
        return {i: SimpleKV() for i in range(len(self.blocks))
                if self.cfg.is_full_attention(i)}

    def new_states(self):
        dev = next(iter(self.buffers())).device if any(True for _ in self.buffers()) \
            else torch.device("cpu")
        return {i: GDNState(self.cfg, dev) for i in range(len(self.blocks))
                if not self.cfg.is_full_attention(i)}

    def set_alpha(self, alpha: float):
        """Change ablation strength at runtime. No reload, no weight edit."""
        self.alpha = float(alpha)

    def n_ablation_sites(self) -> int:
        if self.direction is None or self.alpha == 0.0:
            return 0
        return 2 * len(self.blocks) + 1
