"""End-to-end converter tests against a synthesised GGUF.

We cannot download the real 5.9 GB model here, so we build a small but structurally
genuine GGUF -- correct magic, KV block, tensor directory, alignment and PTQ1_0 payload
-- and push it through the real converter. That exercises every step the real file will
hit except sheer size.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bonsai import codec, format as fmt            # noqa: E402
from tools import convert as conv                  # noqa: E402

GGML_PTQ1_0 = 143
GGML_PQ2_0 = 142


# ------------------------------------------------------------------ minimal writer

def _s(x: bytes) -> bytes:
    return struct.pack("<Q", len(x)) + x


def write_gguf(path: Path, tensors: dict, kv: dict | None = None,
               ggml_type: int = GGML_PTQ1_0) -> dict:
    """``tensors``: name -> (rows, width). Returns the codes/scales actually written."""
    kv = kv or {"general.architecture": "qwen35"}
    rng = np.random.default_rng(99)
    truth, blobs = {}, {}
    for name, (rows, width) in tensors.items():
        c = rng.integers(0, 3, size=(rows, width), dtype=np.int64).astype(np.uint8)
        s = (rng.random((rows, width // codec.GROUP)) * 0.05 + 1e-3).astype(np.float16)
        truth[name] = (c, s)
        blobs[name] = (codec.encode_ptq1_0(c, s) if ggml_type == GGML_PTQ1_0
                       else codec.encode_pq2_0(c, s))

    head = b"GGUF" + struct.pack("<I", 3)
    head += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kv))
    for k, v in kv.items():
        head += _s(k.encode()) + struct.pack("<I", 8) + _s(v.encode())

    offset, infos = 0, b""
    for name, (rows, width) in tensors.items():
        infos += _s(name.encode()) + struct.pack("<I", 2)
        infos += struct.pack("<Q", width) + struct.pack("<Q", rows)   # ne = [in, out]
        infos += struct.pack("<I", ggml_type) + struct.pack("<Q", offset)
        offset += len(blobs[name])
    body = head + infos
    pad = (-len(body)) % 32
    with open(path, "wb") as f:
        f.write(body + b"\0" * pad)
        for name in tensors:
            f.write(blobs[name])
    return truth


# -------------------------------------------------------------------------- fixtures

def tiny_cfg():
    """A miniature but architecturally *self-consistent* Bonsai.

    Every proportion that the converter checks is preserved: the doubled q_proj for the
    output gate, nv != nk in the GDN, the 4-layer full-attention stride, and input dims
    divisible by the 1024-weight superblock. Only the magnitudes shrink, so the real
    code path -- including the shape check -- runs unmodified.
    """
    from bonsai.config import BonsaiConfig
    return BonsaiConfig(
        hidden_size=1024, num_layers=4, intermediate_size=2048, vocab_size=2048,
        num_attention_heads=4, num_key_value_heads=1, head_dim=256,
        linear_num_key_heads=4, linear_num_value_heads=8,
    )


CFG = tiny_cfg()
_TO_GGUF = {}
for _k, _v in conv.LAYER_MAP.items():
    for _i in range(CFG.num_layers):
        _TO_GGUF[_v.format(i=_i)] = _k.format(i=_i) + ".weight"
for _k, _v in conv.GLOBAL_MAP.items():
    _TO_GGUF[_v] = _k + ".weight"

#: a representative slice: both layer types, both ablation-site kinds, embed and head
SAMPLE = ["model.embed_tokens", "lm_head",
          "model.layers.0.linear_attn.in_proj_qkv",
          "model.layers.0.linear_attn.out_proj",
          "model.layers.0.mlp.down_proj",
          "model.layers.3.self_attn.q_proj",
          "model.layers.3.self_attn.o_proj"]
TENSORS = {_TO_GGUF[p]: (o, i) for p, o, i in CFG.packed_modules() if p in SAMPLE}


@pytest.fixture
def gguf(tmp_path):
    p = tmp_path / "tiny-PTQ1_0.gguf"
    truth = write_gguf(p, TENSORS)
    return p, truth


# ----------------------------------------------------------------------------- tests

def test_name_mapping_covers_every_architecture_module():
    """Every one of the 402 packed paths must be reachable from some GGUF name."""
    from bonsai.config import DEFAULT
    reachable = set()
    for i in range(DEFAULT.num_layers):
        for k in conv.LAYER_MAP:
            n = k.format(i=i) + ".weight"
            c = conv.canonical_name(n)
            if c:
                reachable.add(c)
    for k in conv.GLOBAL_MAP:
        reachable.add(conv.canonical_name(k + ".weight"))

    expected = {p for p, _, _ in DEFAULT.packed_modules()}
    missing = expected - reachable
    assert missing == set(), f"no GGUF name maps to: {sorted(missing)[:8]}"


def test_inspect_reports_structure(gguf, capsys):
    path, _ = gguf
    report = conv.inspect(str(path))
    assert report["n_tensors"] == len(TENSORS)
    assert report["types"]["PTQ1_0"] == len(TENSORS)
    assert report["unmapped_ternary"] == []
    assert report["metadata"]["general.architecture"] == "qwen35"
    assert "all ternary tensors mapped" in capsys.readouterr().out


def test_inspect_flags_unmapped_tensors(tmp_path):
    p = tmp_path / "odd.gguf"
    write_gguf(p, {"blk.0.mystery_proj.weight": (128, 5120)})
    assert conv.inspect(str(p))["unmapped_ternary"] == ["blk.0.mystery_proj.weight"]


def test_convert_is_lossless(gguf, tmp_path):
    path, truth = gguf
    out = tmp_path / "out.bonsai"
    conv.convert(str(path), str(out), cfg=CFG, verify=True)

    r = fmt.BonsaiReader(out)
    assert len(r) == len(TENSORS)
    for gguf_name, (rows, width) in TENSORS.items():
        canon = conv.canonical_name(gguf_name)
        assert canon in r
        codes, scales = truth[gguf_name]
        got_c, got_s = fmt.decode_superblocks(*r.ternary(canon), rows, width)
        np.testing.assert_array_equal(got_c, codes)
        np.testing.assert_array_equal(got_s, scales)
        # and the Tier-1 oracle reconstruction
        np.testing.assert_array_equal(r.dequantize(canon),
                                      codec.dequantize(codes, scales))


def test_convert_refuses_pq2_0(tmp_path):
    """PQ2_0 does not leave room for resident KV on 8 GB. Fail loudly, not silently."""
    p = tmp_path / "wrong.gguf"
    write_gguf(p, {"blk.0.ffn_up.weight": (128, 5120)}, ggml_type=GGML_PQ2_0)
    with pytest.raises(SystemExit, match="PTQ1_0"):
        conv.convert(str(p), str(tmp_path / "x.bonsai"), cfg=CFG)


def test_convert_refuses_unmapped_names(tmp_path):
    p = tmp_path / "odd.gguf"
    write_gguf(p, {"blk.0.mystery_proj.weight": (128, 5120)})
    with pytest.raises(SystemExit, match="could not be mapped"):
        conv.convert(str(p), str(tmp_path / "x.bonsai"), cfg=CFG)


def test_convert_detects_shape_disagreement(tmp_path):
    """A plausible-but-wrong name map must be caught by the shape check."""
    p = tmp_path / "bad.gguf"
    write_gguf(p, {"blk.0.ffn_down.weight": (256, 1024)})   # tiny down_proj in is 2048
    with pytest.raises(SystemExit, match="name map is wrong"):
        conv.convert(str(p), str(tmp_path / "x.bonsai"), cfg=CFG)
