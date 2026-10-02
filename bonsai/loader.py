"""Build a runnable model from a ``.bonsai`` container.

Streams tensor by tensor: each packed matrix moves to the device still packed, so peak
host memory stays near one tensor rather than the 5.5 GiB total, and the dense weights
(norms, conv, A_log, dt_bias) follow in fp32.

Nothing is dequantised at load time. 26.9 B parameters at fp16 is 54 GB; the model
holds 1.75 bpw codes and expands each matrix on use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT, BonsaiConfig
from .format import BonsaiReader
from .model import (BonsaiAttention, BonsaiBlock, BonsaiForCausalLM, BonsaiGatedDelta,
                    BonsaiMLP, RuntimeConfig)
from .torch_ops import DenseLinear, TernaryLinear


def _t(a: np.ndarray, device, dtype=None) -> torch.Tensor:
    # np.array(copy=True): container tensors are read-only mmap views, and torch
    # refuses to take ownership of those without warning.
    t = torch.from_numpy(np.array(a, copy=True))
    return t.to(device=device, dtype=dtype) if dtype else t.to(device)


class ModelLoader:
    def __init__(self, path: str | Path, cfg: BonsaiConfig = DEFAULT,
                 rt: RuntimeConfig | None = None):
        self.reader = BonsaiReader(path)
        self.cfg = cfg
        self.rt = rt or RuntimeConfig()
        self.device = torch.device(self.rt.device)

    # ------------------------------------------------------------------ helpers
    def _packed(self, name: str, hadamard: bool = True) -> TernaryLinear:
        codes, scales = self.reader.ternary(name)
        out_f, in_f = self.reader.records[name].shape
        return TernaryLinear(
            out_f, in_f,
            _t(codes.reshape(out_f, -1), self.device),
            _t(scales.reshape(out_f, -1), self.device, torch.float16),
            hadamard=hadamard, cache_weight=self.rt.cache_weights)

    def _dense(self, name: str, shape=None) -> torch.Tensor:
        a = self.reader.dense(name)
        if shape is not None:
            a = a.reshape(shape)
        return _t(a, self.device, torch.float32)

    def _has(self, name: str) -> bool:
        return name in self.reader

    # -------------------------------------------------------------------- build
    def build(self) -> BonsaiForCausalLM:
        cfg, rt = self.cfg, self.rt
        model = BonsaiForCausalLM(cfg, rt)

        model.embed_tokens = self._packed("model.embed_tokens", hadamard=False)
        model.lm_head = self._packed("lm_head", hadamard=True)
        model.norm = self._dense("model.norm")

        for i in range(cfg.num_layers):
            p = f"model.layers.{i}"
            input_ln = self._dense(f"{p}.input_layernorm")
            post_ln = self._dense(f"{p}.post_attention_layernorm")

            if cfg.is_full_attention(i):
                mixer = BonsaiAttention(
                    cfg,
                    self._packed(f"{p}.self_attn.q_proj"),
                    self._packed(f"{p}.self_attn.k_proj"),
                    self._packed(f"{p}.self_attn.v_proj"),
                    self._packed(f"{p}.self_attn.o_proj"),
                    q_norm=(self._dense(f"{p}.self_attn.q_norm")
                            if self._has(f"{p}.self_attn.q_norm") else None),
                    k_norm=(self._dense(f"{p}.self_attn.k_norm")
                            if self._has(f"{p}.self_attn.k_norm") else None))
            else:
                mixer = BonsaiGatedDelta(
                    cfg,
                    self._packed(f"{p}.linear_attn.in_proj_qkv"),
                    self._packed(f"{p}.linear_attn.in_proj_z"),
                    DenseLinear(self._dense(f"{p}.linear_attn.in_proj_a",
                                            (cfg.linear_num_value_heads,
                                             cfg.hidden_size))),
                    DenseLinear(self._dense(f"{p}.linear_attn.in_proj_b",
                                            (cfg.linear_num_value_heads,
                                             cfg.hidden_size))),
                    self._packed(f"{p}.linear_attn.out_proj"),
                    self._dense(f"{p}.linear_attn.conv1d"),
                    self._dense(f"{p}.linear_attn.A_log"),
                    self._dense(f"{p}.linear_attn.dt_bias"),
                    self._dense(f"{p}.linear_attn.norm"))

            mlp = BonsaiMLP(self._packed(f"{p}.mlp.gate_proj"),
                            self._packed(f"{p}.mlp.up_proj"),
                            self._packed(f"{p}.mlp.down_proj"))
            model.blocks.append(BonsaiBlock(cfg, i, mixer, mlp, input_ln, post_ln))

        return model.to(self.device)


def load_model(path: str | Path, direction: str | Path | None = None,
               alpha: float = 1.0, device: str = "cuda",
               dtype: torch.dtype = torch.float16,
               cfg: BonsaiConfig = DEFAULT, **kw) -> BonsaiForCausalLM:
    """Load a ``.bonsai`` model and install the runtime refusal ablation."""
    rt = RuntimeConfig(device=device, dtype=dtype, alpha=alpha, **kw)
    model = ModelLoader(path, cfg, rt).build()
    if direction is not None:
        from .ablation import load_direction
        d = load_direction(direction, cfg.hidden_size)
        model.direction = torch.from_numpy(d).to(model.rope_cos.device)
    return model
