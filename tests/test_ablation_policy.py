"""Ablation policy tests.

These are the port's acceptance criteria, lifted from ``scripts/selfcheck.py`` and
``scripts/test_ablation.py`` and expressed against the native implementation.
"""
from __future__ import annotations

import numpy as np
import pytest

from bonsai import ablation
from bonsai.config import DEFAULT

F32 = np.float32


@pytest.fixture(scope="module")
def policy():
    return ablation.from_repo(alpha=1.0)


# ------------------------------------------------------------------ the direction

def test_shipped_direction_is_unit_norm_and_5120_dim(policy):
    assert policy.direction.shape == (5120,)
    np.testing.assert_allclose(np.linalg.norm(policy.direction), 1.0, atol=1e-6)


def test_both_direction_files_agree():
    raw = ablation.load_direction("directions/refusal_dir_fp32.bin")
    try:
        st = ablation.load_direction("directions/refusal_dir.safetensors")
    except ImportError:
        pytest.skip("safetensors not installed")
    np.testing.assert_allclose(raw, st, atol=1e-6)


def test_load_direction_rejects_wrong_width(tmp_path):
    p = tmp_path / "bad.bin"
    np.arange(100, dtype="<f4").tofile(p)
    with pytest.raises(ValueError, match="expected 5120"):
        ablation.load_direction(p)


# ----------------------------------------------------------------------- the sites

def test_exactly_129_sites_with_the_documented_breakdown(policy):
    d = policy.describe()
    assert d["n_sites"] == 129
    assert d["breakdown"] == {"mlp.down_proj": 64, "linear_attn.out_proj": 48,
                              "self_attn.o_proj": 16, "embed_tokens": 1}


def test_lm_head_is_packed_but_never_a_site(policy):
    """It reads the residual stream and emits logits; it does not write back."""
    assert "lm_head" in {p for p, _, _ in DEFAULT.packed_modules()}
    assert "lm_head" not in policy.sites()


def test_layer_restriction_selects_only_those_layers(policy):
    p = ablation.AblationPolicy(policy.direction, layers={0, 3},
                                include_embedding=False)
    sites = p.sites()
    assert all(int(s.split(".")[2]) in {0, 3} for s in sites)
    # layer 0 is linear-attention (out_proj + down_proj), layer 3 full (o_proj + down)
    assert len(sites) == 4


def test_embedding_can_be_excluded(policy):
    p = ablation.AblationPolicy(policy.direction, include_embedding=False)
    assert len(p.sites()) == 128
    assert DEFAULT.embedding_path not in p.sites()


def test_full_attention_layers_are_exactly_i_mod_4_eq_3(policy):
    o = [s for s in policy.sites() if s.endswith("self_attn.o_proj")]
    assert [int(s.split(".")[2]) for s in o] == list(range(3, 64, 4))


# ------------------------------------------------------------------- the operator

def test_projection_removes_the_component(policy):
    rng = np.random.default_rng(0)
    y = rng.standard_normal((4, 5120)).astype(F32)
    h = policy.apply(y)
    assert max(policy.residual_fraction(row) for row in h) < 1e-6


def test_alpha_zero_is_bit_identical(policy):
    """Required by selfcheck: disabling ablation must not perturb the base model."""
    rng = np.random.default_rng(1)
    y = rng.standard_normal((3, 5120)).astype(np.float16)
    x = rng.standard_normal((3, 5120)).astype(np.float16)
    off = ablation.AblationPolicy(policy.direction, alpha=0.0)
    np.testing.assert_array_equal(off.apply(y), y)
    np.testing.assert_array_equal(
        off.residual_add(x, y),
        (x.astype(F32) + y.astype(F32)).astype(np.float16))


@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.9, 1.0, 2.0])
def test_alpha_scales_the_correction_linearly(policy, alpha):
    rng = np.random.default_rng(2)
    y = rng.standard_normal((2, 5120)).astype(F32)
    p = ablation.AblationPolicy(policy.direction, alpha=alpha)
    expected = y - alpha * (y @ policy.direction)[:, None] * policy.direction
    np.testing.assert_allclose(p.apply(y), expected, rtol=1e-5, atol=1e-6)


def test_equivalent_to_permanent_weight_orthogonalisation(policy):
    """The whole premise: the runtime op equals W <- W - r r^T W, with no requantisation."""
    rng = np.random.default_rng(3)
    W = (rng.standard_normal((5120, 48)) * 0.02).astype(F32)
    v = rng.standard_normal(48).astype(F32)
    r = policy.direction
    np.testing.assert_allclose(
        policy.apply(W @ v), (W - np.outer(r, r @ W)) @ v, rtol=1e-4, atol=1e-5)


def test_fused_residual_add_matches_the_unfused_form(policy):
    """What the CUDA kernel computes must equal add-then-project."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 5120)).astype(F32)
    y = rng.standard_normal((3, 5120)).astype(F32)
    np.testing.assert_allclose(policy.residual_add(x, y), x + policy.apply(y),
                               rtol=1e-5, atol=1e-5)


def test_over_projection_flips_the_component_sign(policy):
    """alpha=2 reflects the component rather than removing it.

    y is given a deliberate component along r: for a random vector the dot product is
    near zero, and measuring a sign flip through that cancellation tests float noise
    rather than the operator.
    """
    rng = np.random.default_rng(5)
    y = (rng.standard_normal((1, 5120)).astype(F32)
         + 3.0 * policy.direction[None, :])
    before = (y @ policy.direction).item()
    assert abs(before) > 1.0
    after = (ablation.AblationPolicy(policy.direction, alpha=2.0).apply(y)
             @ policy.direction).item()
    np.testing.assert_allclose(after, -before, rtol=1e-3)


def test_fp32_accumulation_survives_fp16_activations(policy):
    """R4: with 129 sites and 64 layers, a low-precision reduction would drift."""
    rng = np.random.default_rng(6)
    y = (rng.standard_normal((1, 5120)) * 8).astype(np.float16)
    h = policy.apply(y)
    assert policy.residual_fraction(h) < 2e-3


def test_direction_is_renormalised_if_given_unnormalised(policy):
    p = ablation.AblationPolicy(policy.direction * 7.0)
    np.testing.assert_allclose(np.linalg.norm(p.direction), 1.0, atol=1e-6)
