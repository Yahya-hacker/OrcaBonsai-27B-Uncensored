"""Prove the CUDA kernels' arithmetic on CPU, before any GPU exists.

The kernels in ``bonsai/kernels/*.cu`` cannot be compiled here. But the two things most
likely to be silently wrong in them are pure index/bit arithmetic, and that we *can*
check: this module reimplements the kernel logic line for line in numpy and asserts it
reproduces the reference.

Specifically:

``trit_index``
    the (byte, trit) -> element map that lets the GEMV consume weights out of order.
    Off by one here and the model is fluent and wrong (R6).

the incremental decode
    the kernel carries ``p = (p * 3) & 255`` between trits instead of recomputing
    ``b * 3**t``. Algebraically equal, but worth proving rather than believing.

the fused ablation
    ``h = x + y - alpha*dot(y,r)*r`` must match ``bonsai_abliterate/ablation.py``,
    including the fp32 accumulation, and must be exactly a no-op at alpha == 0.
"""
from __future__ import annotations

import numpy as np
import pytest

from bonsai import codec, format as fmt

GROUP = codec.GROUP
STAGES = codec.PTQ1_0_STAGES


# ------------------------------------------------- mirror of the .cu device functions

def trit_index(byte: int, trit: int) -> int:
    if byte < 16:
        return trit * 16 + byte
    if byte < 24:
        return 80 + trit * 8 + (byte - 16)
    return 120 + trit * 2 + (byte - 24)


def trits_in_byte(byte: int) -> int:
    return 5 if byte < 24 else 4


def kernel_decode_group(block26: np.ndarray) -> np.ndarray:
    """Exactly what one warp does to one 26-byte group, including the carried p."""
    codes = np.zeros(GROUP, dtype=np.uint8)
    for b in range(26):                       # lane
        p = int(block26[b])
        for t in range(trits_in_byte(b)):
            codes[trit_index(b, t)] = ((p & 255) * 3) >> 8
            p = (p * 3) & 255
    return codes


# ----------------------------------------------------------------------------- tests

def test_trit_index_is_a_bijection_onto_the_group():
    seen = [trit_index(b, t) for b in range(26) for t in range(trits_in_byte(b))]
    assert sorted(seen) == list(range(GROUP))


def test_trit_index_matches_the_codec_stage_layout():
    """Derive the map independently from PTQ1_0_STAGES and compare."""
    expected, pos = {}, 0
    for lo, hi, count in STAGES:
        nbytes = hi - lo
        for t in range(count):
            for i in range(nbytes):
                expected[(lo + i, t)] = pos + t * nbytes + i
        pos += nbytes * count
    got = {(b, t): trit_index(b, t) for b in range(26) for t in range(trits_in_byte(b))}
    assert got == expected


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kernel_decode_matches_reference_codec(seed):
    """The warp's incremental decode must equal the authoritative decoder, bit for bit."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 3, size=(4, GROUP), dtype=np.int64).astype(np.uint8)
    scales = np.full((4, 1), 0.02, dtype=np.float16)
    raw = codec.encode_ptq1_0(codes, scales)
    blocks = np.frombuffer(raw, np.uint8).reshape(4, 28)
    for i in range(4):
        np.testing.assert_array_equal(kernel_decode_group(blocks[i, :26]), codes[i])


def test_gemv_simulation_matches_dense_reference():
    """Full GEMV: simulate the kernel over a real-shaped tile and compare to x @ W.T."""
    rng = np.random.default_rng(7)
    rows, in_features = 6, 2048
    codes = rng.integers(0, 3, size=(rows, in_features), dtype=np.int64).astype(np.uint8)
    scales = (rng.random((rows, in_features // GROUP)) * 0.04 + 1e-3).astype(np.float16)
    raw = codec.encode_ptq1_0(codes, scales)
    sb_codes, sb_scales = fmt.split_ptq1_0(raw, rows, in_features)
    x = rng.standard_normal(in_features).astype(np.float32)

    # ---- what the kernel computes
    supers = in_features // fmt.SUPERBLOCK_WEIGHTS
    y = np.zeros(rows, dtype=np.float32)
    for row in range(rows):
        crow = sb_codes[row * supers:(row + 1) * supers]
        acc = 0.0
        for sb in range(supers):
            for g in range(fmt.GROUPS_PER_SUPERBLOCK):
                gidx = sb * fmt.GROUPS_PER_SUPERBLOCK + g
                chunk = crow[sb][g * 26:(g + 1) * 26]
                c = kernel_decode_group(chunk).astype(np.float32) - 1.0
                part = float(c @ x[gidx * GROUP:(gidx + 1) * GROUP])
                acc += part * float(sb_scales[row, gidx])
        y[row] = acc

    reference = codec.dequantize(codes, scales) @ x
    np.testing.assert_allclose(y, reference, rtol=2e-5, atol=2e-4)


# ------------------------------------------------------------------- fused ablation

def ablate(x, y, r, alpha):
    """numpy mirror of residual_add_ablate, fp32 interior."""
    yf, xf = y.astype(np.float32), x.astype(np.float32)
    dot = (yf * r).sum(axis=-1, keepdims=True)
    return (xf + yf - alpha * dot * r).astype(y.dtype)


@pytest.fixture
def direction():
    rng = np.random.default_rng(11)
    r = rng.standard_normal(5120).astype(np.float32)
    return r / np.linalg.norm(r)


def test_ablation_removes_the_component(direction):
    rng = np.random.default_rng(12)
    y = rng.standard_normal((3, 5120)).astype(np.float32)
    x = np.zeros_like(y)
    h = ablate(x, y, direction, 1.0)
    residual = np.abs(h @ direction) / np.linalg.norm(h, axis=-1)
    assert residual.max() < 1e-6, residual


def test_alpha_zero_is_bit_identical(direction):
    """scripts/selfcheck.py depends on this: alpha=0 must not perturb the base model."""
    rng = np.random.default_rng(13)
    y = rng.standard_normal((4, 5120)).astype(np.float16)
    x = rng.standard_normal((4, 5120)).astype(np.float16)
    np.testing.assert_array_equal(ablate(x, y, direction, 0.0),
                                  (x.astype(np.float32) + y.astype(np.float32)).astype(np.float16))


def test_ablation_matches_matrix_orthogonalisation(direction):
    """The runtime op must equal the permanent weight edit W <- W - r r^T W."""
    rng = np.random.default_rng(14)
    W = rng.standard_normal((5120, 64)).astype(np.float32) * 0.02
    v = rng.standard_normal(64).astype(np.float32)
    runtime = ablate(np.zeros(5120, np.float32), W @ v, direction, 1.0)
    baked = (W - np.outer(direction, direction @ W)) @ v
    np.testing.assert_allclose(runtime, baked, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 2.0])
def test_ablation_is_linear_in_alpha(direction, alpha):
    rng = np.random.default_rng(15)
    y = rng.standard_normal((2, 5120)).astype(np.float32)
    x = np.zeros_like(y)
    h = ablate(x, y, direction, alpha)
    expected = y - alpha * (y @ direction)[:, None] * direction
    np.testing.assert_allclose(h, expected, rtol=1e-5, atol=1e-6)


# --------------------------------------------------- fwht.cu butterfly index math

def kernel_fwht(vec: np.ndarray) -> np.ndarray:
    """Mirror of rmsnorm_sign_fwht's butterfly loop, including the lo/hi index split.

    The kernel gives thread ``tid`` butterfly ``tid`` at every stage and derives the
    pair as ``lo = (tid // h) * 2h + (tid % h)``, ``hi = lo + h``. If that split is
    wrong the transform is still *a* linear map -- plausible numbers, wrong model --
    so it is worth checking against a dense Hadamard matrix.
    """
    n = len(vec)
    buf = vec.astype(np.float64).copy()
    h = 1
    while h < n:
        new = buf.copy()
        for tid in range(n // 2):
            lo = (tid // h) * 2 * h + (tid % h)
            hi = lo + h
            a, b = buf[lo], buf[hi]
            new[lo], new[hi] = a + b, a - b
        buf = new
        h <<= 1
    return (buf / np.sqrt(n)).astype(np.float32)


@pytest.mark.parametrize("n", [8, 64, 1024])
def test_kernel_butterfly_indexing_touches_every_pair_once(n):
    h = 1
    while h < n:
        pairs = [((tid // h) * 2 * h + (tid % h)) for tid in range(n // 2)]
        pairs += [p + h for p in pairs]
        assert sorted(pairs) == list(range(n)), f"stage h={h} does not partition"
        h <<= 1


@pytest.mark.parametrize("n", [8, 64, 1024])
def test_kernel_fwht_matches_dense_hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    rng = np.random.default_rng(21)
    x = rng.standard_normal(n).astype(np.float32)
    np.testing.assert_allclose(kernel_fwht(x), (H @ x) / np.sqrt(n),
                               rtol=1e-4, atol=1e-4)


def test_kernel_fwht_matches_the_reference_implementation():
    from bonsai import reference as ref
    rng = np.random.default_rng(22)
    x = rng.standard_normal(1024).astype(np.float32)
    np.testing.assert_allclose(kernel_fwht(x), ref.fwht(x[None, :], 1024)[0],
                               rtol=1e-4, atol=1e-4)


# ----------------------------------------------- gdn_step.cu head sharing + decay

def test_gdn_kernel_head_sharing_matches_the_reference():
    """khead = head // (nv/nk): 3 value heads per key head. Off here and 2/3 of the
    heads read the wrong key."""
    from bonsai.config import DEFAULT
    nv, nk = DEFAULT.linear_num_value_heads, DEFAULT.linear_num_key_heads
    kernel = [h // (nv // nk) for h in range(nv)]
    reference = np.repeat(np.arange(nk), nv // nk).tolist()
    assert kernel == reference
    assert len(set(kernel)) == nk


def test_gdn_kernel_softplus_branch_is_continuous():
    """The kernel uses x for x>20 and log1p(exp(x)) below; the seam must not jump."""
    for x in (19.999, 20.0, 20.001):
        sp = x if x > 20.0 else np.log1p(np.exp(x))
        assert abs(sp - np.logaddexp(x, 0.0)) < 1e-5


# ---------------------------------------------------- kv_quant.cu nibble ordering

def test_kv_kernel_nibble_packing_matches_numpy():
    """Lane i owns byte i = elements 2i and 2i+1, low nibble first."""
    from bonsai import kvcache
    rng = np.random.default_rng(23)
    x = rng.standard_normal((1, 64)).astype(np.float32)
    codes, scale, zero = kvcache.quantize(x, group=64, symmetric=False)

    s, z = float(scale[0, 0]), float(zero[0, 0])
    expected = np.clip(np.rint((x[0] - z) / s), 0, 15).astype(np.uint8)
    got = np.empty(64, np.uint8)
    for i in range(32):                      # the kernel's per-lane byte assembly
        byte = codes[0, 0, i]
        got[2 * i] = byte & 0x0F
        got[2 * i + 1] = byte >> 4
    np.testing.assert_array_equal(got, expected)
