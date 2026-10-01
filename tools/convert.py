#!/usr/bin/env python3
"""Convert a Ternary Bonsai 2 27B GGUF into the native ``.bonsai`` container.

    python tools/convert.py inspect  model-PTQ1_0.gguf [-o report.json]
    python tools/convert.py convert  model-PTQ1_0.gguf -o bonsai-27b.bonsai

``inspect`` is the one to run first. It needs no GPU, reads only the header, finishes in
under a second on a 5.9 GB file, and prints everything the converter needs to know:
tensor names, shapes, ggml type ids and the architecture metadata. Run it before
converting -- if the name map below does not match your file, ``convert`` refuses to
guess and tells you exactly which names it could not place.

Design notes
------------
* **Streaming.** One tensor is in memory at a time. Peak RSS stays a few hundred MB on a
  5.9 GB model, which matters because the target machine has other things to do.
* **Lossless by construction.** Ternary tensors are regrouped, never re-encoded, so a
  trit cannot change. ``--verify`` proves it by reading the result back and comparing
  against the source bytes.
* **The permutation is explicit.** GGUF stores GDN value-head tensors in a different
  head grouping than the runtime wants (``nv=48 != nk=16``). Which tensors need the
  reorder -- and the asymmetric fact that ``ssm_out`` does **not** -- is encoded in
  ``NEEDS_VALUE_PERM`` rather than left to chance. See ``bonsai/config.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bonsai import codec, format as fmt                       # noqa: E402
from bonsai.config import DEFAULT, BonsaiConfig               # noqa: E402
from bonsai_abliterate.gguf_min import Gguf                   # noqa: E402

# --------------------------------------------------------------------- name mapping
# GGUF (llama.cpp qwen35) -> our canonical HF-style path. Per-layer entries take {i}.
LAYER_MAP = {
    "blk.{i}.ffn_gate":   "model.layers.{i}.mlp.gate_proj",
    "blk.{i}.ffn_up":     "model.layers.{i}.mlp.up_proj",
    "blk.{i}.ffn_down":   "model.layers.{i}.mlp.down_proj",
    # full attention
    "blk.{i}.attn_q":     "model.layers.{i}.self_attn.q_proj",
    "blk.{i}.attn_k":     "model.layers.{i}.self_attn.k_proj",
    "blk.{i}.attn_v":     "model.layers.{i}.self_attn.v_proj",
    "blk.{i}.attn_output": "model.layers.{i}.self_attn.o_proj",
    # gated delta net
    "blk.{i}.attn_qkv":   "model.layers.{i}.linear_attn.in_proj_qkv",
    "blk.{i}.attn_gate":  "model.layers.{i}.linear_attn.in_proj_z",
    "blk.{i}.ssm_beta":   "model.layers.{i}.linear_attn.in_proj_b",
    "blk.{i}.ssm_alpha":  "model.layers.{i}.linear_attn.in_proj_a",
    "blk.{i}.ssm_out":    "model.layers.{i}.linear_attn.out_proj",
}
GLOBAL_MAP = {
    "token_embd": "model.embed_tokens",
    "output":     "lm_head",
}

#: GDN tensors whose value-head grouping differs between GGUF and the runtime.
#: ``ssm_out`` is deliberately absent -- its *input* dim needs no permutation, measured
#: at 2.08e-04 identity error against 1.38 for either permutation.
NEEDS_VALUE_PERM = ("attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta",
                    "ssm_a", "ssm_dt", "ssm_conv1d")


def canonical_name(gguf_name: str) -> str | None:
    stem = gguf_name[:-7] if gguf_name.endswith(".weight") else gguf_name
    if stem in GLOBAL_MAP:
        return GLOBAL_MAP[stem]
    parts = stem.split(".")
    if len(parts) >= 3 and parts[0] == "blk" and parts[1].isdigit():
        key = f"blk.{{i}}.{'.'.join(parts[2:])}"
        if key in LAYER_MAP:
            return LAYER_MAP[key].format(i=parts[1])
    return None


# -------------------------------------------------------------------------- inspect

def inspect(path: str, out: str | None = None) -> dict:
    g = Gguf(path)
    by_type, by_suffix, unmapped = Counter(), Counter(), []
    tensors = []
    for name, t in g.tensors.items():
        tname = t["type_name"]
        by_type[tname] += 1
        stem = name[:-7] if name.endswith(".weight") else name
        suffix = ".".join(p for p in stem.split(".") if not p.isdigit())
        by_suffix[suffix] += 1
        canon = canonical_name(name)
        if canon is None and tname in ("PQ2_0", "PTQ1_0"):
            unmapped.append(name)
        tensors.append({"name": name, "shape": list(t["ne"]), "type": tname,
                        "canonical": canon})

    arch_kv = {k: v for k, v in g.kv.items()
               if not k.startswith("tokenizer.") and not isinstance(v, list)}
    report = {
        "path": path, "gguf_version": g.version, "data_start": g.data_start,
        "n_tensors": len(g.tensors),
        "types": dict(by_type), "suffixes": dict(by_suffix),
        "unmapped_ternary": unmapped, "metadata": arch_kv, "tensors": tensors,
    }

    print(f"{path}\n  GGUF v{g.version}, {len(g.tensors)} tensors, "
          f"data at {g.data_start}")
    print("\n  tensor types")
    for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"    {k:>10}  {v:>5}")
    print("\n  tensor families")
    for k, v in sorted(by_suffix.items()):
        print(f"    {k:<34} {v:>4}")
    print("\n  architecture metadata")
    for k, v in sorted(arch_kv.items()):
        print(f"    {k:<42} {v}")
    n_tern = sum(v for k, v in by_type.items() if k in ("PQ2_0", "PTQ1_0"))
    print(f"\n  ternary tensors: {n_tern} (expected {len(DEFAULT.packed_modules())})")
    if unmapped:
        print(f"  !! {len(unmapped)} ternary tensors have no name mapping:")
        for n in unmapped[:12]:
            print(f"       {n}")
        if len(unmapped) > 12:
            print(f"       ... and {len(unmapped)-12} more")
        print("     -> add them to LAYER_MAP / GLOBAL_MAP in tools/convert.py")
    else:
        print("  all ternary tensors mapped")

    if out:
        Path(out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\n  wrote {out}")
    return report


# -------------------------------------------------------------------------- convert

def convert(src: str, dst: str, cfg: BonsaiConfig = DEFAULT,
            verify: bool = False, limit: int | None = None) -> None:
    g = Gguf(src)
    expected = {p: (o, i) for p, o, i in cfg.packed_modules()}

    plan, unmapped = [], []
    for name, t in g.tensors.items():
        if t["type_name"] not in ("PQ2_0", "PTQ1_0"):
            continue
        canon = canonical_name(name)
        (plan.append((name, canon, t)) if canon else unmapped.append(name))
    if unmapped:
        raise SystemExit(
            f"{len(unmapped)} ternary tensors could not be mapped, e.g. "
            f"{unmapped[:5]}.\nRun `inspect` and extend LAYER_MAP in tools/convert.py. "
            "Refusing to guess -- a wrong name map produces a model that is fluent and "
            "wrong.")

    fmts = {t["type_name"] for _, _, t in plan}
    if fmts != {"PTQ1_0"}:
        raise SystemExit(
            f"this converter writes the 1.75 bpw layout and needs a PTQ1_0 source; "
            f"found {sorted(fmts)}. On an 8 GB card PQ2_0 does not leave room for a "
            f"resident KV cache -- download the *-PTQ1_0.gguf.")

    if limit:
        plan = plan[:limit]
    print(f"converting {len(plan)} ternary tensors -> {dst}")

    meta = {"source": Path(src).name, "arch": g.kv.get("general.architecture"),
            "layout": fmt.LAYOUT_PTQ_VERBATIM, "bpw": 1.75,
            "config": {"hidden_size": cfg.hidden_size, "num_layers": cfg.num_layers}}
    done = 0
    with fmt.BonsaiWriter(dst, meta=meta) as w:
        for name, canon, t in plan:
            rows, width = g.shape_out_in(name)
            if canon in expected and (rows, width) != expected[canon]:
                raise SystemExit(
                    f"{name} -> {canon}: GGUF says {(rows, width)}, architecture says "
                    f"{expected[canon]}. The name map is wrong.")
            raw = g.raw(name)
            codes, scales = fmt.split_ptq1_0(raw, rows, width)
            if verify and fmt.restore_ptq1_0(codes, scales, rows, width) != raw:
                raise SystemExit(f"{name}: regroup was not byte-exact")
            w.add_ternary(canon, codes, scales, (rows, width))
            done += 1
            if done % 25 == 0 or done == len(plan):
                print(f"  {done}/{len(plan)}  {canon}")

    size = Path(dst).stat().st_size
    print(f"wrote {dst}  {size/1024**3:.2f} GiB")
    if limit is None and done != len(expected):
        print(f"  note: wrote {done} ternary tensors, architecture expects "
              f"{len(expected)}. Dense tensors (norms, conv1d, A_log, dt_bias, "
              f"in_proj_a/b) are not yet carried -- see M0 notes.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("inspect", help="dump GGUF structure (no GPU, ~1s)")
    pi.add_argument("gguf"); pi.add_argument("-o", "--out")
    pc = sub.add_parser("convert", help="GGUF -> .bonsai")
    pc.add_argument("gguf"); pc.add_argument("-o", "--out", required=True)
    pc.add_argument("--verify", action="store_true",
                    help="prove every tensor round-trips byte-exactly (slower)")
    pc.add_argument("--limit", type=int, help="convert only the first N (smoke test)")
    a = ap.parse_args()
    if a.cmd == "inspect":
        inspect(a.gguf, a.out)
    else:
        convert(a.gguf, a.out, verify=a.verify, limit=a.limit)


if __name__ == "__main__":
    main()
