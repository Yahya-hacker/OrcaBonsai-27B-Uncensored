"""Verified architecture description for Ternary Bonsai 2 27B.

Every constant here was derived from the pack's ``config.json``, the GGUF headers, or
the bundled runtime -- not from prose. The derivation self-checks three ways and the
test suite asserts all three:

* packed modules sum to **402**, matching the pack manifest and the count of ggml
  type-142/143 tensors in the GGUF;
* parameters sum to **26.90 B**, matching the published ``llama-bench`` line;
* residual writers sum to **129**, matching ``directions/direction.json`` and the 258
  tensors (129 pairs) in ``gguf/bonsai-abliterate-lora.gguf``.

If a refactor breaks one of those, the model is wrong, not the test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GIB = 1024 ** 3
MIB = 1024 ** 2


@dataclass(frozen=True)
class BonsaiConfig:
    # ---- global
    hidden_size: int = 5120
    num_layers: int = 64
    intermediate_size: int = 17408
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    eos_token_id: int = 248044
    bos_token_id: int = 248044

    # ---- full attention (layers where i % full_attention_interval == 3)
    full_attention_interval: int = 4
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    attn_output_gate: bool = True
    partial_rotary_factor: float = 0.25
    rope_theta: float = 1e7
    mrope_interleaved: bool = True
    mrope_section: tuple = (11, 11, 10)

    # ---- gated delta net (linear attention)
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    ssm_state_dtype: str = "float32"          # pack config: mamba_ssm_dtype

    # ---- quantisation
    group_size: int = 128
    hadamard_block: int = 1024

    # ---- ablation
    residual_writer_suffixes: tuple = (
        "mlp.down_proj", "self_attn.o_proj", "linear_attn.out_proj",
    )
    embedding_path: str = "model.embed_tokens"

    # --------------------------------------------------------------- derived geometry
    @property
    def key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim        # 2048

    @property
    def value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim    # 6144

    @property
    def conv_dim(self) -> int:
        return self.key_dim * 2 + self.value_dim                           # 10240

    @property
    def attn_qkv_out(self) -> int:
        """q_proj width. Doubled because its second half is the output gate."""
        return self.num_attention_heads * self.head_dim * 2                # 12288

    @property
    def attn_kv_out(self) -> int:
        return self.num_key_value_heads * self.head_dim                    # 1024

    @property
    def attn_inner(self) -> int:
        return self.num_attention_heads * self.head_dim                    # 6144

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)             # 64

    def is_full_attention(self, layer: int) -> bool:
        return layer % self.full_attention_interval == self.full_attention_interval - 1

    def layer_types(self) -> list[str]:
        return ["full_attention" if self.is_full_attention(i) else "linear_attention"
                for i in range(self.num_layers)]

    # ---------------------------------------------------------------- module inventory
    def packed_modules(self) -> list[tuple[str, int, int]]:
        """``(path, out_features, in_features)`` for every ternary-packed matrix."""
        h, out = self.hidden_size, []
        out.append((self.embedding_path, self.vocab_size, h))
        for i in range(self.num_layers):
            p = f"model.layers.{i}"
            if self.is_full_attention(i):
                out += [
                    (f"{p}.self_attn.q_proj", self.attn_qkv_out, h),
                    (f"{p}.self_attn.k_proj", self.attn_kv_out, h),
                    (f"{p}.self_attn.v_proj", self.attn_kv_out, h),
                    (f"{p}.self_attn.o_proj", h, self.attn_inner),
                ]
            else:
                out += [
                    (f"{p}.linear_attn.in_proj_qkv", self.conv_dim, h),
                    (f"{p}.linear_attn.in_proj_z", self.value_dim, h),
                    (f"{p}.linear_attn.out_proj", h, self.value_dim),
                ]
            out += [
                (f"{p}.mlp.gate_proj", self.intermediate_size, h),
                (f"{p}.mlp.up_proj", self.intermediate_size, h),
                (f"{p}.mlp.down_proj", h, self.intermediate_size),
            ]
        out.append(("lm_head", self.vocab_size, h))
        return out

    def residual_writers(self) -> list[str]:
        """The 129 modules whose output is added to the residual stream.

        ``lm_head`` is packed but is deliberately *not* a writer: it reads the residual
        stream and produces logits.
        """
        return [p for p, _, _ in self.packed_modules()
                if p == self.embedding_path or p.endswith(self.residual_writer_suffixes)]

    def unpacked_modules(self) -> list[tuple[str, int]]:
        """``(path, numel)`` for tensors kept at full precision.

        Small but not negligible: including them is what closes the gap between the
        402 packed matrices (26.87 B) and the published 26.90 B total.
        """
        h = self.hidden_size
        out = [("model.norm", h)]
        for i in range(self.num_layers):
            p = f"model.layers.{i}"
            out += [(f"{p}.input_layernorm", h), (f"{p}.post_attention_layernorm", h)]
            if self.is_full_attention(i):
                out += [(f"{p}.self_attn.q_norm", self.head_dim),
                        (f"{p}.self_attn.k_norm", self.head_dim)]
            else:
                nv = self.linear_num_value_heads
                out += [
                    # in_proj_a / in_proj_b are too narrow to be worth packing
                    (f"{p}.linear_attn.in_proj_a", h * nv),
                    (f"{p}.linear_attn.in_proj_b", h * nv),
                    (f"{p}.linear_attn.conv1d", self.conv_dim * self.linear_conv_kernel_dim),
                    (f"{p}.linear_attn.A_log", nv),
                    (f"{p}.linear_attn.dt_bias", nv),
                    (f"{p}.linear_attn.norm", self.linear_value_head_dim),
                ]
        return out

    def num_parameters(self, include_unpacked: bool = True) -> int:
        n = sum(o * i for _, o, i in self.packed_modules())
        if include_unpacked:
            n += sum(k for _, k in self.unpacked_modules())
        return n

    # ------------------------------------------------------------------ memory budget
    def weight_bytes(self, bpw: float = 1.75) -> int:
        return int(self.num_parameters() * bpw / 8)

    def gdn_state_bytes(self, dtype_bytes: int = 4) -> int:
        n_linear = self.num_layers - self.num_layers // self.full_attention_interval
        return (n_linear * self.linear_num_value_heads
                * self.linear_value_head_dim * self.linear_key_head_dim * dtype_bytes)

    def kv_bytes_per_token(self, bits: int = 16) -> int:
        n_full = self.num_layers // self.full_attention_interval
        return n_full * self.num_key_value_heads * self.head_dim * 2 * bits // 8

    def memory_plan(self, vram_gib: float = 8.0, usable_frac: float = 0.95,
                    bpw: float = 1.75, kv_bits: int = 4,
                    overhead_gib: float = 0.65, state_dtype_bytes: int = 4) -> dict:
        """Budget the card. Returns the maximum fully-resident context.

        ``overhead_gib`` covers the CUDA context, cuBLAS workspace, activation scratch
        and allocator fragmentation -- measure it and pass the real number.
        """
        usable = vram_gib * GIB * usable_frac
        weights = self.weight_bytes(bpw)
        state = self.gdn_state_bytes(state_dtype_bytes)
        fixed = weights + state + overhead_gib * GIB
        free = usable - fixed
        per_token = self.kv_bytes_per_token(kv_bits)
        return {
            "usable_bytes": usable,
            "weight_bytes": weights,
            "gdn_state_bytes": state,
            "overhead_bytes": overhead_gib * GIB,
            "fixed_bytes": fixed,
            "kv_free_bytes": free,
            "kv_bytes_per_token": per_token,
            "max_resident_context": int(free // per_token) if free > 0 else 0,
        }

    def decode_bytes_per_token(self, context: int, bpw: float = 1.75,
                               kv_bits: int = 4, state_dtype_bytes: int = 4) -> int:
        """Bytes the decode step must move. Decode is bandwidth-bound, so this over
        memory bandwidth is the speed ceiling."""
        return (self.weight_bytes(bpw)
                + 2 * self.gdn_state_bytes(state_dtype_bytes)   # read + write
                + context * self.kv_bytes_per_token(kv_bits))

    def decode_roofline(self, bandwidth_gbs: float, context: int,
                        efficiency: float = 0.55, **kw) -> float:
        """Projected tokens/sec. ``efficiency`` 0.43-0.55 matches llama.cpp on this
        model family; 1.0 gives the hard ceiling."""
        per_tok = self.decode_bytes_per_token(context, **kw)
        return bandwidth_gbs * 1e9 * efficiency / per_tok


# ------------------------------------------------------------------ GGUF import quirks

def value_head_permutation(unit: int, num_value_heads: int = 48,
                           num_key_heads: int = 16) -> np.ndarray:
    """The V-head reorder required when importing GDN tensors **from GGUF**.

    ``nv=48 != nk=16``, so the GGUF value-head grouping has to be transposed into the
    runtime's. The MLX pack already has this applied -- ``PACK-RUNTIME.md`` warns "do
    not permute them again".

    Applies to ``attn_qkv`` (value part), ``attn_gate``, ``ssm_alpha``, ``ssm_beta``,
    ``ssm_a``, ``ssm_dt.bias`` and ``ssm_conv1d`` (value part). It does **not** apply to
    ``ssm_out``'s input dimension -- ``export_gguf_lora.py --check`` measured identity
    at 2.08e-04 against 1.38 for either permutation. Getting this backwards produces a
    model that is fluent and wrong.
    """
    repeat = num_value_heads // num_key_heads
    return (np.arange(num_value_heads * unit)
            .reshape(repeat, num_key_heads, unit)
            .transpose(1, 0, 2)
            .reshape(-1))


DEFAULT = BonsaiConfig()
