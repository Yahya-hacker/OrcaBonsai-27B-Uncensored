"""Container round-trip tests: conversion must be provably lossless."""
from __future__ import annotations

import numpy as np
import pytest

from bonsai import codec, format as fmt
from bonsai.config import DEFAULT, value_head_permutation


def _synth(rng, rows, width):
    codes = rng.integers(0, 3, size=(rows, width), dtype=np.int64).astype(np.uint8)
    scales = (rng.random((rows, width // codec.GROUP)) * 0.05 + 1e-3).astype(np.float16)
    return codes, scales


SHAPES = [(1, 1024), (4, 5120), (7, 6144), (3, 17408)]


@pytest.mark.parametrize("rows,width", SHAPES)
def test_split_restore_is_byte_exact(rows, width):
    """Regrouping must not alter a single byte of the source GGUF block."""
    rng = np.random.default_rng(1)
    codes, scales = _synth(rng, rows, width)
    raw = codec.encode_ptq1_0(codes, scales)
    c, s = fmt.split_ptq1_0(raw, rows, width)
    assert raw == fmt.restore_ptq1_0(c, s, rows, width)


@pytest.mark.parametrize("rows,width", SHAPES)
def test_superblock_decode_matches_source(rows, width):
    rng = np.random.default_rng(2)
    codes, scales = _synth(rng, rows, width)
    raw = codec.encode_ptq1_0(codes, scales)
    c, s = fmt.split_ptq1_0(raw, rows, width)
    got_codes, got_scales = fmt.decode_superblocks(c, s, rows, width)
    np.testing.assert_array_equal(got_codes, codes)
    np.testing.assert_array_equal(got_scales, scales)


def test_superblock_density_is_exactly_1_75_bpw():
    bits = (fmt.SUPERBLOCK_CODE_BYTES + fmt.SUPERBLOCK_SCALE_BYTES) * 8
    assert bits / fmt.SUPERBLOCK_WEIGHTS == 1.75
    assert fmt.SUPERBLOCK_CODE_BYTES % 16 == 0      # uint4 loads


def test_every_packed_module_is_superblock_divisible():
    """All 402 input dims must divide by 1024, or the container cannot hold them."""
    bad = [(p, i) for p, _, i in DEFAULT.packed_modules()
           if i % fmt.SUPERBLOCK_WEIGHTS]
    assert bad == []


def test_container_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    path = tmp_path / "t.bonsai"
    rows, width = 8, 5120
    codes, scales = _synth(rng, rows, width)
    raw = codec.encode_ptq1_0(codes, scales)
    c, s = fmt.split_ptq1_0(raw, rows, width)
    norm = rng.standard_normal(5120).astype(np.float32)

    with fmt.BonsaiWriter(path, meta={"hello": "world"}) as w:
        w.add_ternary("blk.0.ffn_down", c, s, (rows, width))
        w.add_dense("blk.0.norm", norm)

    r = fmt.BonsaiReader(path)
    assert len(r) == 2 and r.meta["hello"] == "world"
    gc, gs = r.ternary("blk.0.ffn_down")
    np.testing.assert_array_equal(gc, c)
    np.testing.assert_array_equal(gs, s)
    np.testing.assert_array_equal(r.dense("blk.0.norm"), norm)

    # the Tier-1 oracle path
    ref = codec.dequantize(codes, scales)
    np.testing.assert_array_equal(r.dequantize("blk.0.ffn_down"), ref)


def test_container_rejects_unaligned_width():
    rng = np.random.default_rng(4)
    codes, scales = _synth(rng, 2, 640)          # 640 = 5 groups, not a superblock
    raw = codec.encode_ptq1_0(codes, scales)
    with pytest.raises(ValueError):
        fmt.split_ptq1_0(raw, 2, 640)


# ------------------------------------------------------------------ the R1 permutation

def test_value_head_permutation_is_an_involution_free_bijection():
    perm = value_head_permutation(128)
    assert perm.shape == (48 * 128,)
    assert sorted(perm.tolist()) == list(range(48 * 128))


def test_value_head_permutation_regroups_heads_not_elements():
    """head h of the GGUF layout must land contiguously, and not be the identity."""
    unit = 4
    perm = value_head_permutation(unit, 48, 16)
    assert not np.array_equal(perm, np.arange(48 * unit))
    # element j of head g should stay element j of whatever head it moves to
    assert np.array_equal(perm.reshape(-1, unit) % unit,
                          np.tile(np.arange(unit), (48, 1)))
