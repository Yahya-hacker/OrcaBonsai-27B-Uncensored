"""Bit-exactness tests for the ternary codecs.

The important test here is :func:`test_matches_pack_reference`. It checks our decoder
against a verbatim copy of the pack's own ``runtime/codec.py`` -- an independent
implementation written by the model's authors. Agreeing with ourselves proves nothing;
agreeing with them is the thing that keeps the port honest.

Runs on CPU with numpy only. No GPU, no model files.
"""
from __future__ import annotations

import numpy as np
import pytest

from bonsai import codec


# ---------------------------------------------------------------- the pack's own code
# Verbatim from prism-ml/Ternary-Bonsai-2-27B-mlx-2bit : runtime/codec.py
# Kept byte-for-byte so it stays an independent oracle. Do not "clean this up".

def _pack_transcode(raw: bytes, shape, source: str):
    rows, width = shape
    sizes = {"PQ2_0": 34, "PTQ1_0": 28}
    blocks = rows * width // 128
    data = np.frombuffer(raw, dtype=np.uint8).reshape(blocks, sizes[source])
    scale_bytes = data[:, :2] if source == "PQ2_0" else data[:, 26:28]
    scales = scale_bytes.copy().view("<f2").reshape(rows, width // 128)
    if source == "PQ2_0":
        words = data[:, 2:].copy().view("<u4").reshape(rows, width // 16)
        return words, np.ascontiguousarray(scales), np.ascontiguousarray(-scales)
    pieces = []
    for lo, hi, count in [(0, 16, 5), (16, 24, 5), (24, 26, 4)]:
        packed = data[:, lo:hi].astype(np.uint16)
        for trit in range(count):
            remainder = (packed * (3 ** trit)) & 255
            pieces.append(((remainder * 3) >> 8).astype(np.uint8))
    codes = np.concatenate(pieces, axis=1)
    words = np.bitwise_or.reduce(
        codes.astype(np.uint32).reshape(rows, width // 16, 16)
        << (2 * np.arange(16, dtype=np.uint32)),
        axis=-1,
    ).astype("<u4")
    return (np.ascontiguousarray(words),
            np.ascontiguousarray(scales),
            np.ascontiguousarray(-scales))


def _pack_unpack(weight, scales, biases):
    rows, words = weight.shape
    values = np.empty((rows, words * 16), dtype=np.float32)
    for lane in range(16):
        values[:, lane::16] = (weight >> (2 * lane)) & 3
    groups = values.reshape(rows, -1, 128)
    return (groups * scales.astype(np.float32)[..., None]
            + biases.astype(np.float32)[..., None]).reshape(rows, -1)


# ----------------------------------------------------------------------------- fixtures

def _random_block(rng, rows, width):
    codes = rng.integers(0, 3, size=(rows, width), dtype=np.int64).astype(np.uint8)
    scales = (rng.random((rows, width // codec.GROUP)) * 0.05 + 1e-3).astype(np.float16)
    return codes, scales


SHAPES = [(1, 128), (3, 256), (8, 1024), (17, 5120)]


# -------------------------------------------------------------------------------- tests

@pytest.mark.parametrize("count", [4, 5])
def test_trit_table_is_surjective(count):
    """Every one of the 3**count trit tuples must be reachable, or encoding is lossy."""
    table = codec._trit_table(count)
    tuples = {tuple(row) for row in table}
    assert len(tuples) == 3 ** count


@pytest.mark.parametrize("count", [4, 5])
def test_trit_table_inverse_round_trips(count):
    inv = codec._TRIT_INV[count]
    fwd = codec._trit_table(count)
    weights = 3 ** np.arange(count)
    for idx in range(3 ** count):
        byte = inv[idx]
        assert (fwd[byte] * weights).sum() == idx


@pytest.mark.parametrize("rows,width", SHAPES)
def test_ptq1_0_round_trip(rows, width):
    rng = np.random.default_rng(0xB04)
    codes, scales = _random_block(rng, rows, width)
    raw = codec.encode_ptq1_0(codes, scales)
    assert len(raw) == rows * width // codec.GROUP * 28
    got, got_scales = codec.decode_ptq1_0(raw, rows, width)
    np.testing.assert_array_equal(got, codes)
    np.testing.assert_array_equal(got_scales, scales)


@pytest.mark.parametrize("rows,width", SHAPES)
def test_pq2_0_round_trip(rows, width):
    rng = np.random.default_rng(0xB05)
    codes, scales = _random_block(rng, rows, width)
    raw = codec.encode_pq2_0(codes, scales)
    assert len(raw) == rows * width // codec.GROUP * 34
    got, got_scales = codec.decode_pq2_0(raw, rows, width)
    np.testing.assert_array_equal(got, codes)
    np.testing.assert_array_equal(got_scales, scales)


@pytest.mark.parametrize("fmt", [codec.PTQ1_0, codec.PQ2_0])
@pytest.mark.parametrize("rows,width", SHAPES)
def test_matches_pack_reference(fmt, rows, width):
    """Our decode must equal the model authors' own transcode + unpack. Bit for bit."""
    rng = np.random.default_rng(0xC0DEC)
    codes, scales = _random_block(rng, rows, width)
    enc = codec.encode_ptq1_0 if fmt == codec.PTQ1_0 else codec.encode_pq2_0
    raw = enc(codes, scales)

    words, ref_scales, ref_biases = _pack_transcode(raw, (rows, width), fmt)
    reference = _pack_unpack(words, ref_scales, ref_biases)

    ours = codec.dequantize(*codec.decode(raw, rows, width, fmt))
    np.testing.assert_array_equal(ours, reference)


def test_bias_identity_gives_clean_ternary_levels():
    """bias == -scale means codes {0,1,2} decode to exactly {-s, 0, +s}."""
    codes = np.array([[0, 1, 2] * 64], dtype=np.uint8)[:, :128]
    scales = np.array([[0.25]], dtype=np.float16)
    w = codec.dequantize(codes, scales)
    assert set(np.unique(w).tolist()) <= {-0.25, 0.0, 0.25}
    np.testing.assert_array_equal(w[0, :3], [-0.25, 0.0, 0.25])


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        codec.decode_ptq1_0(b"\x00" * 28, 1, 100)        # width not a multiple of 128
    with pytest.raises(ValueError):
        codec.decode_ptq1_0(b"\x00" * 27, 1, 128)        # short buffer
