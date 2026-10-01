"""Tests for the Tier-2 reference math.

These target the four documented silent-failure modes (R2-R5) plus the properties the
CUDA kernels will be graded on. A failure here means the oracle itself is wrong, which
would be worse than a wrong kernel -- everything downstream is measured against it.
"""
from __future__ import annotations

import numpy as np
import pytest

from bonsai import reference as ref
from bonsai.config import DEFAULT

F32 = np.float32


# ------------------------------------------------------------------- elementwise

def test_silu_and_sigmoid_against_closed_form():
    x = np.linspace(-8, 8, 101).astype(F32)
    np.testing.assert_allclose(ref.sigmoid(x), 1 / (1 + np.exp(-x)), rtol=1e-6)
    np.testing.assert_allclose(ref.silu(x), x / (1 + np.exp(-x)), rtol=1e-6)


def test_softplus_is_stable_at_extremes():
    x = np.array([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=F32)
    out = ref.softplus(x)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[-1], 100.0, rtol=1e-5)   # no overflow
    assert out[0] >= 0.0


def test_rms_norm_normalises():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 512)).astype(F32) * 7.0
    y = ref.rms_norm(x, None)
    np.testing.assert_allclose(np.sqrt((y ** 2).mean(-1)), 1.0, rtol=1e-4)


def test_rms_norm_gated_is_precise_swiglu_not_a_plain_multiply():
    """R4: gate goes through silu, and the interior is fp32."""
    rng = np.random.default_rng(1)
    h = rng.standard_normal((3, 128)).astype(np.float16)
    g = rng.standard_normal((3, 128)).astype(np.float16)
    w = np.ones(128, dtype=F32)
    got = ref.rms_norm_gated(h, g, w)
    expected = (ref.silu(g) * ref.rms_norm(h, w)).astype(np.float16)
    np.testing.assert_array_equal(got, expected)
    assert not np.allclose(got.astype(F32), (g.astype(F32) * ref.rms_norm(h, w)))


# ---------------------------------------------------------------------- hadamard

@pytest.mark.parametrize("block", [8, 64, 1024])
def test_fwht_is_orthogonal_and_self_inverse(block):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, block * 2)).astype(F32)
    y = ref.fwht(x, block)
    np.testing.assert_allclose((y ** 2).sum(-1), (x ** 2).sum(-1), rtol=1e-4)
    np.testing.assert_allclose(ref.fwht(y, block), x, atol=1e-4)


def test_fwht_matches_dense_hadamard_matrix():
    block = 8
    H = np.array([[1]], dtype=F32)
    while H.shape[0] < block:
        H = np.block([[H, H], [H, -H]])
    x = np.arange(block, dtype=F32)[None, :]
    np.testing.assert_allclose(ref.fwht(x, block), (x @ H.T) / np.sqrt(block), atol=1e-5)


def test_fwht_sign_order_differs_between_forward_and_inverse():
    """R5: the embedding applies signs *after* the transform, not before."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 64)).astype(F32)
    s = rng.choice([-1.0, 1.0], size=64).astype(F32)
    assert not np.allclose(ref.fwht(x, 64, s, inverse=False),
                           ref.fwht(x, 64, s, inverse=True))


# ---------------------------------------------------------------------- ablation

def test_ablate_removes_component_and_alpha_zero_is_identity():
    rng = np.random.default_rng(4)
    r = rng.standard_normal(5120).astype(F32)
    r /= np.linalg.norm(r)
    y = rng.standard_normal((3, 5120)).astype(F32)
    h = ref.ablate(y, r, 1.0)
    assert np.abs(h @ r).max() / np.linalg.norm(h, axis=-1).min() < 1e-6
    np.testing.assert_array_equal(ref.ablate(y, r, 0.0), y)


def test_ablate_uses_the_shipped_direction():
    """The real direction must be unit norm and 5120-dim, as direction.json claims."""
    r = np.fromfile("directions/refusal_dir_fp32.bin", dtype="<f4")
    assert r.shape == (DEFAULT.hidden_size,)
    np.testing.assert_allclose(np.linalg.norm(r), 1.0, atol=1e-6)
    rng = np.random.default_rng(5)
    y = rng.standard_normal((2, 5120)).astype(F32)
    h = ref.ablate(y, r, 1.0)
    assert np.abs(h @ r).max() < 1e-3


# --------------------------------------------------------------------- attention

def test_split_q_and_gate_widths_match_the_architecture():
    """R2: q_proj is 24*256*2 = 12288, half queries and half gate."""
    cfg = DEFAULT
    assert cfg.attn_qkv_out == 12288
    x = np.zeros((7, cfg.attn_qkv_out), dtype=F32)
    q, gate = ref.split_q_and_gate(x, cfg.num_attention_heads, cfg.head_dim)
    assert q.shape == (7, 24, 256)
    assert gate.shape == (7, 6144) == (7, cfg.attn_inner)


def test_output_gate_is_sigmoid_not_silu():
    rng = np.random.default_rng(6)
    a = rng.standard_normal((2, 64)).astype(F32)
    g = rng.standard_normal((2, 64)).astype(F32)
    np.testing.assert_allclose(ref.apply_output_gate(a, g), a * ref.sigmoid(g), rtol=1e-6)
    assert not np.allclose(ref.apply_output_gate(a, g), a * ref.silu(g))


def test_output_gate_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        ref.apply_output_gate(np.zeros((2, 64)), np.zeros((2, 32)))


def test_attention_is_causal_and_normalised():
    rng = np.random.default_rng(7)
    q = rng.standard_normal((4, 6, 32)).astype(F32)
    k = rng.standard_normal((2, 6, 32)).astype(F32)       # GQA 4/2
    v = rng.standard_normal((2, 6, 32)).astype(F32)
    out = ref.attention(q, k, v)
    assert out.shape == (4, 6, 32)
    # token 0 may only see itself, so its output equals v[head0, 0]
    np.testing.assert_allclose(out[0, 0], np.repeat(v, 2, axis=0)[0, 0], rtol=1e-5)


def test_rope_rotates_only_the_partial_slice():
    """R7: 64 of 256 dims at partial_rotary_factor 0.25."""
    cfg = DEFAULT
    assert cfg.rotary_dim == 64
    rng = np.random.default_rng(8)
    x = rng.standard_normal((5, cfg.head_dim)).astype(F32)
    y = ref.rope(x, np.arange(5), cfg.rotary_dim, cfg.rope_theta)
    np.testing.assert_array_equal(y[:, cfg.rotary_dim:], x[:, cfg.rotary_dim:])
    assert not np.allclose(y[1:, :cfg.rotary_dim], x[1:, :cfg.rotary_dim])


def test_rope_preserves_norm_and_is_identity_at_position_zero():
    rng = np.random.default_rng(9)
    x = rng.standard_normal((3, 256)).astype(F32)
    y = ref.rope(x, np.zeros(3), 64)
    np.testing.assert_allclose(y, x, atol=1e-6)
    y = ref.rope(x, np.arange(3), 64)
    np.testing.assert_allclose(np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1),
                               rtol=1e-5)


# --------------------------------------------------------------- gated delta net

def test_gdn_qk_scaling_is_asymmetric():
    """R3: inv**2 on q, inv on k. If these ever become equal, the model changed."""
    rng = np.random.default_rng(10)
    q = rng.standard_normal((16, 128)).astype(F32)
    k = rng.standard_normal((16, 128)).astype(F32)
    qs, ks = ref.gdn_qk_scale(q, k)
    inv = 128 ** -0.5
    np.testing.assert_allclose(qs, inv * inv * ref.rms_norm(q, None), rtol=1e-5)
    np.testing.assert_allclose(ks, inv * ref.rms_norm(k, None), rtol=1e-5)
    ratio = np.linalg.norm(qs) / np.linalg.norm(ks)
    np.testing.assert_allclose(ratio, inv, rtol=1e-4)


def test_gdn_decay_is_in_unit_interval():
    rng = np.random.default_rng(11)
    a = rng.standard_normal(48).astype(F32) * 3
    g = ref.gdn_decay(a, np.zeros(48, F32), np.zeros(48, F32))
    assert ((g > 0) & (g <= 1)).all()


def test_gdn_step_shapes_and_head_sharing():
    cfg = DEFAULT
    nv, nk = cfg.linear_num_value_heads, cfg.linear_num_key_heads
    dv, dk = cfg.linear_value_head_dim, cfg.linear_key_head_dim
    rng = np.random.default_rng(12)
    state = np.zeros((nv, dv, dk), F32)
    q = rng.standard_normal((nk, dk)).astype(F32)
    k = rng.standard_normal((nk, dk)).astype(F32)
    v = rng.standard_normal((nv, dv)).astype(F32)
    g = np.ones(nv, F32)
    beta = np.full(nv, 0.5, F32)
    s, y = ref.gdn_step(state, q, k, v, g, beta)
    assert s.shape == (nv, dv, dk) and y.shape == (nv, dv)
    assert np.isfinite(s).all() and np.isfinite(y).all()


def test_gdn_state_is_context_independent_in_size():
    """48 of 64 layers cost nothing as context grows -- the reason 27B fits at all."""
    cfg = DEFAULT
    assert cfg.gdn_state_bytes(4) == 144 * 1024 ** 2
    assert cfg.gdn_state_bytes(2) == 72 * 1024 ** 2      # README's fp16 figure


def test_gdn_delta_rule_writes_then_reads():
    """With g=1, beta=1 and q==k on a zero state, y must recover v."""
    rng = np.random.default_rng(13)
    nv, dv, dk = 4, 8, 8
    state = np.zeros((nv, dv, dk), F32)
    k = rng.standard_normal((nv, dk)).astype(F32)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    v = rng.standard_normal((nv, dv)).astype(F32)
    _, y = ref.gdn_step(state, k, k, v, np.ones(nv, F32), np.ones(nv, F32))
    np.testing.assert_allclose(y, v, rtol=1e-4, atol=1e-5)


def test_gdn_decay_erases_state():
    rng = np.random.default_rng(14)
    nv, dv, dk = 2, 4, 4
    state = rng.standard_normal((nv, dv, dk)).astype(F32)
    z = np.zeros((nv, dk), F32)
    s, y = ref.gdn_step(state, z, z, np.zeros((nv, dv), F32),
                        np.zeros(nv, F32), np.zeros(nv, F32))
    np.testing.assert_allclose(s, 0.0, atol=1e-6)


# ------------------------------------------------------------------------- conv1d

def test_causal_conv1d_matches_manual_window():
    rng = np.random.default_rng(15)
    L, C, K = 6, 3, 4
    x = rng.standard_normal((L, C)).astype(F32)
    w = rng.standard_normal((C, K)).astype(F32)
    out, state = ref.causal_conv1d(x, w)
    assert state.shape == (K - 1, C)
    padded = np.concatenate([np.zeros((K - 1, C), F32), x])
    for t in range(L):
        np.testing.assert_allclose(out[t], (padded[t:t + K] * w.T).sum(0), rtol=1e-5)


def test_causal_conv1d_state_carries_across_calls():
    """Streaming one token at a time must equal processing the sequence at once."""
    rng = np.random.default_rng(16)
    L, C, K = 7, 5, 4
    x = rng.standard_normal((L, C)).astype(F32)
    w = rng.standard_normal((C, K)).astype(F32)
    whole, _ = ref.causal_conv1d(x, w)
    state, chunks = None, []
    for t in range(L):
        y, state = ref.causal_conv1d(x[t:t + 1], w, state)
        chunks.append(y)
    np.testing.assert_allclose(np.concatenate(chunks), whole, rtol=1e-5, atol=1e-6)
