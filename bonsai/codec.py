"""Ternary block codecs for Bonsai 2 27B: PQ2_0, PTQ1_0, and our native layout.

This module is the project's ground truth for bit layout. Every kernel is validated
against the reconstruction produced here, so if this file is wrong, nothing downstream
can be right -- and it will be wrong *silently*, producing fluent nonsense rather than
an error. It is therefore written to mirror the pack's own ``runtime/codec.py``
arithmetic exactly, and the test suite proves the round trip.

Two source formats
------------------

``PQ2_0``  34 bytes per 128 weights (2.125 bpw)
    ``[0:2]``   FP16 group scale
    ``[2:34]``  128 x 2-bit codes, 16 per little-endian uint32

``PTQ1_0``  28 bytes per 128 weights (1.75 bpw)  <-- what we ship
    ``[0:26]``  128 base-3 trits, 5 per byte, in three stages
    ``[26:28]`` FP16 group scale

The trit packing is not a plain base-3 integer. A byte ``b`` encodes the first ``k``
digits of the base-3 expansion of the *fraction* ``b / 256``::

    trit_t = ((b * 3**t) & 255) * 3 >> 8

Over the 256 possible bytes this is surjective onto all 3**5 = 243 trit tuples (proved
in ``tests/test_codec.py``) but not injective, so several bytes decode to the same
tuple. Decoding is therefore exact and canonical; *encoding* must pick a representative,
which this module does via an inverted lookup table. We only ever encode to synthesise
test vectors -- the converter copies source bytes verbatim.

The affine container
--------------------

All 402 packed modules satisfy ``bias == -scale`` exactly, so the stored codes
``{0, 1, 2}`` decode to ``{-s, 0, +s}``::

    w = (code - 1) * scale

We bake that identity in and never materialise a bias plane, which is where the pack's
redundant 0.391 GiB goes.
"""
from __future__ import annotations

import numpy as np

GROUP = 128
"""Weights per quantisation group. Fixed by the trained scales -- changing it would
require requantisation, which is exactly what this project refuses to do."""

PQ2_0 = "PQ2_0"
PTQ1_0 = "PTQ1_0"

BLOCK_BYTES = {PQ2_0: 34, PTQ1_0: 28}
GGML_TYPE = {142: PQ2_0, 143: PTQ1_0}

# GGML PTQ stages: (byte_lo, byte_hi, trits_per_byte). 16*5 + 8*5 + 2*4 = 128.
PTQ1_0_STAGES = ((0, 16, 5), (16, 24, 5), (24, 26, 4))


# --------------------------------------------------------------------------- tables

def _trit_table(count: int) -> np.ndarray:
    """``(256, count)`` uint8 table of the trits each byte value decodes to."""
    b = np.arange(256, dtype=np.uint16)[:, None]
    t = np.arange(count, dtype=np.uint16)[None, :]
    return (((b * (3 ** t)) & 255) * 3 >> 8).astype(np.uint8)


def _inverse_trit_table(count: int) -> np.ndarray:
    """``(3**count,)`` uint8 table mapping a packed base-3 index back to a byte.

    The forward map is surjective but not injective; we take the smallest byte that
    produces each tuple so encoding is deterministic.
    """
    fwd = _trit_table(count)
    weights = (3 ** np.arange(count)).astype(np.int64)
    keys = (fwd.astype(np.int64) * weights).sum(axis=1)
    inv = np.full(3 ** count, 255, dtype=np.uint8)
    # iterate high->low so the lowest byte wins
    for byte in range(255, -1, -1):
        inv[keys[byte]] = byte
    return inv


_TRIT = {c: _trit_table(c) for c in (4, 5)}
_TRIT_INV = {c: _inverse_trit_table(c) for c in (4, 5)}


# -------------------------------------------------------------------------- decoding

def _check(raw: bytes | np.ndarray, rows: int, width: int, fmt: str) -> np.ndarray:
    if width % GROUP:
        raise ValueError(f"input dim {width} is not a multiple of {GROUP}")
    blocks = rows * width // GROUP
    need = blocks * BLOCK_BYTES[fmt]
    if len(raw) != need:
        raise ValueError(
            f"{fmt}: expected {need} bytes for [{rows}, {width}], got {len(raw)}"
        )
    return np.frombuffer(raw, dtype=np.uint8).reshape(blocks, BLOCK_BYTES[fmt])


def decode_ptq1_0(raw, rows: int, width: int):
    """``(codes uint8 [rows, width] in {0,1,2}, scales float16 [rows, width/128])``."""
    data = _check(raw, rows, width, PTQ1_0)
    scales = data[:, 26:28].copy().view("<f2").reshape(rows, width // GROUP)
    pieces = []
    for lo, hi, count in PTQ1_0_STAGES:
        pieces.append(_TRIT[count][data[:, lo:hi]].transpose(0, 2, 1).reshape(len(data), -1))
    codes = np.concatenate(pieces, axis=1).reshape(rows, width)
    return codes, np.ascontiguousarray(scales)


def decode_pq2_0(raw, rows: int, width: int):
    """``(codes uint8 [rows, width] in {0,1,2}, scales float16 [rows, width/128])``."""
    data = _check(raw, rows, width, PQ2_0)
    scales = data[:, 0:2].copy().view("<f2").reshape(rows, width // GROUP)
    words = data[:, 2:].copy().view("<u4")                       # (blocks, 8)
    lanes = np.arange(16, dtype=np.uint32)
    codes = ((words[:, :, None] >> (2 * lanes)) & 3).astype(np.uint8)
    return codes.reshape(rows, width), np.ascontiguousarray(scales)


def decode(raw, rows: int, width: int, fmt: str):
    if fmt == PTQ1_0:
        return decode_ptq1_0(raw, rows, width)
    if fmt == PQ2_0:
        return decode_pq2_0(raw, rows, width)
    raise ValueError(f"unsupported ternary format {fmt!r}")


def dequantize(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """``(code - 1) * scale`` in float32 -- the ``bias == -scale`` identity, baked in."""
    rows, width = codes.shape
    s = scales.astype(np.float32).repeat(GROUP, axis=1)
    if s.shape != (rows, width):
        raise ValueError(f"scale shape {scales.shape} does not match codes {codes.shape}")
    return (codes.astype(np.float32) - 1.0) * s


# -------------------------------------------------------------------------- encoding
# Only used to synthesise test vectors; the converter never re-encodes source bytes.

def encode_ptq1_0(codes: np.ndarray, scales: np.ndarray) -> bytes:
    rows, width = codes.shape
    if codes.max(initial=0) > 2:
        raise ValueError("ternary codes must be in {0,1,2}")
    blocks = rows * width // GROUP
    flat = codes.reshape(blocks, GROUP)
    out = np.zeros((blocks, BLOCK_BYTES[PTQ1_0]), dtype=np.uint8)
    pos = 0
    for lo, hi, count in PTQ1_0_STAGES:
        nbytes = hi - lo
        chunk = flat[:, pos:pos + nbytes * count].reshape(blocks, count, nbytes)
        idx = (chunk.astype(np.int64)
               * (3 ** np.arange(count, dtype=np.int64))[None, :, None]).sum(axis=1)
        out[:, lo:hi] = _TRIT_INV[count][idx]
        pos += nbytes * count
    out[:, 26:28] = scales.astype("<f2").reshape(blocks, 1).view(np.uint8)
    return out.tobytes()


def encode_pq2_0(codes: np.ndarray, scales: np.ndarray) -> bytes:
    rows, width = codes.shape
    blocks = rows * width // GROUP
    flat = codes.reshape(blocks, 8, 16).astype(np.uint32)
    words = np.bitwise_or.reduce(flat << (2 * np.arange(16, dtype=np.uint32)), axis=-1)
    out = np.zeros((blocks, BLOCK_BYTES[PQ2_0]), dtype=np.uint8)
    out[:, 0:2] = scales.astype("<f2").reshape(blocks, 1).view(np.uint8)
    out[:, 2:] = words.astype("<u4").view(np.uint8).reshape(blocks, 32)
    return out.tobytes()
