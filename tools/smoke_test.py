#!/usr/bin/env python3
"""Prove the engine works on this machine, without the 5.9 GB model.

    python tools/smoke_test.py            # portable torch path
    python tools/smoke_test.py --cuda     # and the compiled kernels, with parity

Builds a miniature but structurally identical model (4 layers, one of them full
attention, real packed weights in a real .bonsai container), runs prefill, incremental
decode and generation through the real code, and checks:

  1. torch dequantisation is bit-exact against the numpy Tier-1 oracle;
  2. incremental decode equals a single prefill pass -- the KV cache and the gated
     delta state are being carried correctly;
  3. alpha = 0 is bit-identical to the base model;
  4. the ablation actually changes the output, and scales with alpha;
  5. with --cuda, the compiled kernels agree with the portable path.

If this passes, the only thing between you and the 27B model is the weights.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bonsai import codec                                            # noqa: E402
from bonsai.config import BonsaiConfig                              # noqa: E402
from bonsai.format import BonsaiReader, BonsaiWriter, split_ptq1_0  # noqa: E402

OK, BAD = "  [ok]  ", "  [FAIL]"
_fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _fail
    print(f"{OK if cond else BAD} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        _fail += 1


def tiny_cfg() -> BonsaiConfig:
    return BonsaiConfig(
        hidden_size=1024, num_layers=4, intermediate_size=2048, vocab_size=2048,
        max_position_embeddings=4096, num_attention_heads=4, num_key_value_heads=1,
        head_dim=256, linear_num_key_heads=4, linear_num_value_heads=8,
        linear_key_head_dim=128, linear_value_head_dim=128)


def build(path: Path, cfg: BonsaiConfig) -> None:
    rng = np.random.default_rng(0)
    with BonsaiWriter(path, {"arch": "qwen35", "quant": "PTQ1_0"}) as w:
        for name, out_f, in_f in cfg.packed_modules():
            codes = rng.integers(0, 3, (out_f, in_f), dtype=np.uint8)
            scales = rng.random((out_f, in_f // 128)).astype(np.float16) * 0.05 + 0.01
            c, s = split_ptq1_0(codec.encode_ptq1_0(codes, scales), out_f, in_f)
            w.add_ternary(name, c, s, (out_f, in_f))
        for name, size in cfg.unpacked_modules():
            if name.endswith(("layernorm", "model.norm", "linear_attn.norm")):
                a = np.ones(size, np.float32)
            elif name.endswith("dt_bias"):
                a = np.zeros(size, np.float32)
            else:
                a = rng.standard_normal(size).astype(np.float32) * 0.05
            w.add_dense(name, a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true", help="also exercise the GPU path")
    a = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. See docs/BUILDING-CUDA.md.")
        return 2

    from bonsai.generate import SamplingConfig, generate
    from bonsai.loader import ModelLoader
    from bonsai.model import RuntimeConfig
    from bonsai.torch_ops import dequantize_ternary

    device = "cpu"
    if a.cuda:
        if not torch.cuda.is_available():
            print("--cuda given but torch.cuda.is_available() is False. "
                  "Run tools/doctor.py.")
            return 2
        device = "cuda"
        print(f"device: {torch.cuda.get_device_name()} "
              f"(sm_{''.join(map(str, torch.cuda.get_device_capability()))})")
    print(f"torch {torch.__version__} on {device}\n")

    cfg = tiny_cfg()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tiny.bonsai"
        t0 = time.time()
        build(path, cfg)
        print(f"built a {path.stat().st_size / 1e6:.1f} MB container "
              f"({len(cfg.packed_modules())} packed tensors) in {time.time()-t0:.1f}s\n")

        dtype = torch.float32 if device == "cpu" else torch.float16
        model = ModelLoader(path, cfg,
                            RuntimeConfig(device=device, dtype=dtype,
                                          alpha=0.0)).build()

        # 1 -- dequantisation parity against the Tier-1 oracle
        r = BonsaiReader(path)
        name, out_f, in_f = cfg.packed_modules()[0]
        codes, scales = r.ternary(name)
        got = dequantize_ternary(
            torch.from_numpy(np.array(codes, copy=True).reshape(out_f, -1)),
            torch.from_numpy(np.array(scales, copy=True)),
            out_f, in_f, torch.float32).numpy()
        check("torch dequant is bit-exact vs the numpy oracle",
              np.array_equal(got, r.dequantize(name)), name)

        # 2 -- forward
        ids = torch.tensor([[2, 8, 3, 11, 6, 4]], device=device)
        out = model(ids, model.new_caches(), model.new_states())
        check("forward produces finite logits",
              bool(torch.isfinite(out).all()), str(tuple(out.shape)))

        # 3 -- incremental decode == prefill
        seq = [2, 8, 3, 11, 6, 4]
        full = model(torch.tensor([seq], device=device),
                     model.new_caches(), model.new_states())[0, -1]
        c, s = model.new_caches(), model.new_states()
        model(torch.tensor([seq[:-1]], device=device), c, s, offset=0)
        step = model(torch.tensor([[seq[-1]]], device=device), c, s,
                     offset=len(seq) - 1)[0, -1]
        err = (full - step).abs().max().item()
        check("incremental decode == full prefill (cache + GDN state)",
              err < 5e-3, f"max abs diff {err:.2e}")

        # 4 -- the ablation
        d = np.random.default_rng(1).standard_normal(cfg.hidden_size).astype(np.float32)
        d /= np.linalg.norm(d)
        model.direction = torch.from_numpy(d).to(device)
        model.set_alpha(0.0)
        a0 = model(ids, model.new_caches(), model.new_states())
        model.direction = None
        base = model(ids, model.new_caches(), model.new_states())
        check("alpha=0 is bit-identical to the base model",
              bool(torch.equal(a0, base)))

        model.direction = torch.from_numpy(d).to(device)
        model.set_alpha(1.0)
        a1 = model(ids, model.new_caches(), model.new_states())
        model.set_alpha(2.0)
        a2 = model(ids, model.new_caches(), model.new_states())
        check("ablation changes the output", not torch.allclose(a0, a1))
        check("stronger alpha moves it further",
              (a2 - a0).abs().mean() > (a1 - a0).abs().mean())
        model.direction = None
        model.set_alpha(0.0)

        # 5 -- generation
        t0 = time.time()
        toks = list(generate(model, [1, 2, 3],
                             SamplingConfig(temperature=0.0, max_tokens=16),
                             eos_ids=set()))
        dt = time.time() - t0
        check("greedy generation runs", len(toks) == 16,
              f"{len(toks)} tokens in {dt:.2f}s ({len(toks)/max(dt,1e-9):.1f} tok/s)")
        again = list(generate(model, [1, 2, 3],
                              SamplingConfig(temperature=0.0, max_tokens=16),
                              eos_ids=set()))
        check("greedy generation is reproducible", toks == again)

        # 6 -- kernels
        if a.cuda:
            from bonsai.kernels import available, load
            if available():
                k = load()
                x = torch.randn(1, cfg.hidden_size, device="cuda", dtype=torch.float16)
                y = torch.randn_like(x)
                rr = torch.from_numpy(d).cuda()
                al = torch.tensor([1.0], device="cuda")
                fused = k.residual_ablate(x, y, rr, al)
                ref = (x.float() + y.float()
                       - ((y.float() * rr).sum(-1, keepdim=True)) * rr).half()
                e = (fused - ref).abs().max().item()
                check("CUDA fused ablation matches the torch path", e < 2e-2,
                      f"max abs diff {e:.2e}")
            else:
                print("  [skip] compiled kernels unavailable -- "
                      "see docs/BUILDING-CUDA.md")

    print()
    if _fail:
        print(f"{_fail} check(s) FAILED")
        return 1
    print("all checks passed -- the engine works on this machine.")
    print("next: convert the GGUF (tools/convert.py) and run run_cuda.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
