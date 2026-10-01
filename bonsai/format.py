"""The ``.bonsai`` container: a GPU-shaped home for ternary weights.

Why not just read the GGUF directly? Because the GGUF block layout is tuned for CPU
SIMD, and two of its properties are actively hostile to a CUDA kernel:

* **scales are interleaved with codes** (2 bytes every 26 or 32), so a kernel streaming
  weights pays for strided, unaligned loads of both;
* **blocks are 28 bytes**, so nothing lands on a 16-byte boundary and ``uint4`` vector
  loads are unavailable.

This container fixes both without touching a single trit:

* **Separate planes.** Codes and scales live in their own contiguous arrays, so each is
  loaded fully coalesced.
* **1024-weight super-blocks.** Eight 128-weight groups are stored back to back as
  ``8 x 26 = 208`` bytes, which is ``13 x 16`` -- 16-byte aligned, so ``uint4`` loads
  work. 1024 is also the Hadamard block size, so a super-block is exactly one
  transform's worth of input.
* **No bias plane.** ``bias == -scale`` holds exactly for all 402 modules, so the
  identity is baked into the kernel and the pack's redundant 0.391 GiB disappears.

Density is ``(208 + 16) * 8 / 1024 = 1.75`` bits per weight, matching PTQ1_0 exactly.

What this deliberately does *not* do
------------------------------------
It does not re-order trits *within* a group. The 16/8/2-byte stage split is inherited
verbatim from PTQ1_0, which makes conversion a provably lossless regroup -- we never
re-encode, so we cannot introduce a silent encoding bug. A finer interleave (xyz-llama
uses ILV16) may well be faster, but choosing one without a profiler is guesswork. The
container is versioned so M1 can change it behind ``layout_id`` once there is a GPU to
measure on.
"""
from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import codec

MAGIC = b"BONSAI\x00\x00"
VERSION = 1
HEADER_BYTES = 128
DATA_ALIGN = 4096

SUPERBLOCK_WEIGHTS = 1024
GROUPS_PER_SUPERBLOCK = SUPERBLOCK_WEIGHTS // codec.GROUP          # 8
TRIT_BYTES_PER_GROUP = 26                                          # PTQ1_0 minus scale
SUPERBLOCK_CODE_BYTES = GROUPS_PER_SUPERBLOCK * TRIT_BYTES_PER_GROUP   # 208
SUPERBLOCK_SCALE_BYTES = GROUPS_PER_SUPERBLOCK * 2                     # 16

LAYOUT_PTQ_VERBATIM = "ptq1_0-verbatim-sb1024"


@dataclass
class TensorRecord:
    name: str
    kind: str                 # "ternary" | "dense"
    shape: tuple
    dtype: str                # dense only
    layout: str               # ternary only
    code_offset: int = 0
    code_bytes: int = 0
    scale_offset: int = 0
    scale_bytes: int = 0
    data_offset: int = 0      # dense only
    data_bytes: int = 0


def _align(n: int, a: int = DATA_ALIGN) -> int:
    return n + (-n) % a


# ------------------------------------------------------------------------- repacking

def regroup_codes(group_bytes: np.ndarray) -> np.ndarray:
    """``[n_groups, 26] -> [n_superblocks, 208]``. Pure reshape; bytes are untouched."""
    n = len(group_bytes)
    if n % GROUPS_PER_SUPERBLOCK:
        raise ValueError(
            f"{n} groups is not a multiple of {GROUPS_PER_SUPERBLOCK}; the input "
            f"dimension must be divisible by {SUPERBLOCK_WEIGHTS}"
        )
    return group_bytes.reshape(n // GROUPS_PER_SUPERBLOCK, SUPERBLOCK_CODE_BYTES)


def split_ptq1_0(raw: bytes, rows: int, width: int):
    """Split raw PTQ1_0 into ``(codes [n_sb, 208] uint8, scales [rows, width/128] f16)``.

    This is the whole conversion for a ternary tensor: lift the scales out, regroup the
    trit bytes. No arithmetic touches the codes, so it cannot be lossy.
    """
    if width % SUPERBLOCK_WEIGHTS:
        raise ValueError(f"input dim {width} must be divisible by {SUPERBLOCK_WEIGHTS}")
    blocks = rows * width // codec.GROUP
    data = np.frombuffer(raw, dtype=np.uint8).reshape(blocks, codec.BLOCK_BYTES[codec.PTQ1_0])
    scales = data[:, 26:28].copy().view("<f2").reshape(rows, width // codec.GROUP)
    codes = regroup_codes(np.ascontiguousarray(data[:, :26]))
    return codes, np.ascontiguousarray(scales)


def restore_ptq1_0(codes: np.ndarray, scales: np.ndarray, rows: int, width: int) -> bytes:
    """Inverse of :func:`split_ptq1_0`, for round-trip proofs."""
    blocks = rows * width // codec.GROUP
    out = np.empty((blocks, codec.BLOCK_BYTES[codec.PTQ1_0]), dtype=np.uint8)
    out[:, :26] = codes.reshape(blocks, TRIT_BYTES_PER_GROUP)
    out[:, 26:28] = scales.astype("<f2").reshape(blocks, 1).view(np.uint8)
    return out.tobytes()


def decode_superblocks(codes: np.ndarray, scales: np.ndarray, rows: int, width: int):
    """Reconstruct ``[rows, width]`` ternary codes from container planes.

    The reference path every kernel is tested against.
    """
    return codec.decode_ptq1_0(restore_ptq1_0(codes, scales, rows, width), rows, width)


# ----------------------------------------------------------------------------- writer

class BonsaiWriter:
    """Streaming writer -- never holds more than one tensor in memory.

    Matters on the target machine: the source GGUF is 5.9 GB and the box has 32 GB of
    RAM that other things also want.
    """

    def __init__(self, path: str | Path, meta: dict | None = None):
        self.path = Path(path)
        self.meta = dict(meta or {})
        self.records: list[TensorRecord] = []
        self._fh = open(self.path, "wb")
        self._fh.write(b"\0" * HEADER_BYTES)      # placeholder; rewritten on close
        self._cursor = HEADER_BYTES

    def _write(self, buf: np.ndarray | bytes) -> tuple[int, int]:
        raw = buf.tobytes() if isinstance(buf, np.ndarray) else buf
        pad = (-self._cursor) % 16
        if pad:
            self._fh.write(b"\0" * pad)
            self._cursor += pad
        off = self._cursor
        self._fh.write(raw)
        self._cursor += len(raw)
        return off, len(raw)

    def add_ternary(self, name: str, codes: np.ndarray, scales: np.ndarray,
                    shape: tuple) -> None:
        co, cb = self._write(np.ascontiguousarray(codes, dtype=np.uint8))
        so, sb = self._write(np.ascontiguousarray(scales, dtype=np.float16))
        self.records.append(TensorRecord(
            name=name, kind="ternary", shape=tuple(int(s) for s in shape), dtype="",
            layout=LAYOUT_PTQ_VERBATIM,
            code_offset=co, code_bytes=cb, scale_offset=so, scale_bytes=sb))

    def add_dense(self, name: str, array: np.ndarray) -> None:
        a = np.ascontiguousarray(array)
        off, nb = self._write(a)
        self.records.append(TensorRecord(
            name=name, kind="dense", shape=tuple(int(s) for s in a.shape),
            dtype=str(a.dtype), layout="",
            data_offset=off, data_bytes=nb))

    def close(self) -> None:
        directory = json.dumps(
            {"meta": self.meta, "tensors": [asdict(r) for r in self.records]},
            separators=(",", ":")).encode()
        dir_off, dir_len = self._write(directory)
        self._fh.seek(0)
        self._fh.write(MAGIC)
        self._fh.write(struct.pack("<IIQQQ", VERSION, 0, len(self.records),
                                   dir_off, dir_len))
        self._fh.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


# ----------------------------------------------------------------------------- reader

class BonsaiReader:
    """Memory-mapped reader. Tensors are views, so nothing is copied until used."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            head = f.read(HEADER_BYTES)
        if head[:8] != MAGIC:
            raise ValueError(f"{path} is not a .bonsai container")
        version, _flags, n, dir_off, dir_len = struct.unpack("<IIQQQ", head[8:40])
        if version != VERSION:
            raise ValueError(f"container version {version}, expected {VERSION}")
        with open(self.path, "rb") as f:
            f.seek(dir_off)
            blob = json.loads(f.read(dir_len))
        self.meta = blob["meta"]
        self.records = {r["name"]: TensorRecord(**r) for r in blob["tensors"]}
        if len(self.records) != n:
            raise ValueError("directory length disagrees with header")
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")

    def __len__(self): return len(self.records)
    def __contains__(self, name): return name in self.records
    def names(self): return list(self.records)

    def ternary(self, name: str):
        r = self.records[name]
        if r.kind != "ternary":
            raise KeyError(f"{name} is not a ternary tensor")
        rows, width = r.shape
        codes = self._mm[r.code_offset:r.code_offset + r.code_bytes].reshape(
            -1, SUPERBLOCK_CODE_BYTES)
        scales = (self._mm[r.scale_offset:r.scale_offset + r.scale_bytes]
                  .view(np.float16).reshape(rows, width // codec.GROUP))
        return codes, scales

    def dense(self, name: str) -> np.ndarray:
        r = self.records[name]
        if r.kind != "dense":
            raise KeyError(f"{name} is not a dense tensor")
        return (self._mm[r.data_offset:r.data_offset + r.data_bytes]
                .view(np.dtype(r.dtype)).reshape(r.shape))

    def dequantize(self, name: str) -> np.ndarray:
        """Tier-1 oracle: exact fp32 reconstruction of one matrix.

        The largest is 17408x5120 -> 340 MiB in fp32, 178 MiB in fp16. Cheap enough to
        validate every kernel against, one matrix at a time, without ever needing the
        whole model in memory.
        """
        r = self.records[name]
        rows, width = r.shape
        codes, scales = self.ternary(name)
        c, s = decode_superblocks(codes, scales, rows, width)
        return codec.dequantize(c, s)
