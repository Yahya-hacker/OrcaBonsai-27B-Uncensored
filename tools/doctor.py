#!/usr/bin/env python3
"""One command that reports everything the native runtime needs to know about a machine.

    python tools/doctor.py              # report
    python tools/doctor.py -o doc.json  # and save it

Answers, in one pass, the questions the memory and throughput plan currently has to
assume: real memory bandwidth, PCIe generation and width, how much VRAM the display is
already holding, and whether the CUDA toolchain can actually target this GPU. Then it
computes the resident-context budget and the projected decode rate **for this machine**
rather than for a datasheet.

Safe to run anywhere: everything is optional and degrades to "unknown" rather than
raising. No model files, no downloads.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bonsai.config import DEFAULT, GIB          # noqa: E402
from bonsai.kvcache import KVConfig             # noqa: E402

FIELDS = [
    "name", "memory.total", "memory.used", "memory.free",
    "pcie.link.gen.current", "pcie.link.gen.max",
    "pcie.link.width.current", "pcie.link.width.max",
    "clocks.max.memory", "clocks.max.sm", "driver_version", "power.limit",
]


def _run(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None


def gpu_info() -> dict:
    out = _run(["nvidia-smi", f"--query-gpu={','.join(FIELDS)}",
                "--format=csv,noheader,nounits"])
    if not out or not out.strip():
        return {"available": False}
    vals = [v.strip() for v in out.strip().splitlines()[0].split(",")]
    info = dict(zip(FIELDS, vals))
    info["available"] = True
    for k in ("memory.total", "memory.used", "memory.free"):
        try:
            info[k + ".gib"] = round(float(info[k]) / 1024, 2)      # MiB -> GiB
        except Exception:
            pass
    # GDDR7 on a 128-bit bus: bandwidth = clock(MHz) * 2 (DDR) * bus/8
    try:
        clk = float(info["clocks.max.memory"])
        info["est_bandwidth_gbs_128bit"] = round(clk * 2 * 128 / 8 / 1000, 1)
    except Exception:
        pass
    return info


def toolchain_info() -> dict:
    info: dict = {}
    nvcc = _run(["nvcc", "--version"])
    if nvcc:
        for line in nvcc.splitlines():
            if "release" in line:
                info["nvcc"] = line.split("release")[-1].strip().strip(",")
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["capability"] = list(torch.cuda.get_device_capability())
            free, total = torch.cuda.mem_get_info()
            info["torch_free_gib"] = round(free / GIB, 2)
            info["torch_total_gib"] = round(total / GIB, 2)
    except ImportError:
        info["torch"] = None
    return info


def verdicts(gpu: dict, tc: dict) -> list[str]:
    out = []
    cap = tc.get("capability")
    if cap:
        sm = f"sm_{cap[0]}{cap[1]}"
        out.append(f"GPU is {sm}")
        if cap[0] >= 12:
            cu = tc.get("torch_cuda") or "0.0"
            major, minor = (int(x) for x in (cu.split(".") + ["0"])[:2])
            if (major, minor) < (12, 8):
                out.append(
                    f"BLOCKER: torch is built against CUDA {cu}, but {sm} needs 12.8+. "
                    "Kernels will fail at launch with 'no kernel image is available'. "
                    "pip install torch --index-url https://download.pytorch.org/whl/cu128")
            else:
                out.append(f"torch CUDA {cu} supports {sm}")
    if gpu.get("pcie.link.gen.current"):
        gen = gpu["pcie.link.gen.current"]
        width = gpu.get("pcie.link.width.current", "?")
        approx = {"3": 1.0, "4": 2.0, "5": 4.0}.get(str(gen), 0) * int(
            width if str(width).isdigit() else 8) * 0.985
        out.append(f"PCIe gen{gen} x{width} ~ {approx:.0f} GB/s")
        out.append(
            "  at that rate an fp16 KV cache offloaded to host RAM costs "
            f"{32768 * 65536 / max(approx, 1) / 1e9 * 1000:.0f} ms per token at 32K "
            "context -- which is exactly what the resident 4-bit cache avoids")
    free = gpu.get("memory.free.gib") or tc.get("torch_free_gib")
    total = gpu.get("memory.total.gib") or tc.get("torch_total_gib")
    if free and total and total - free > 0.3:
        out.append(f"NOTE: {total - free:.2f} GiB of VRAM is already in use "
                   "(display/compositor). Budget below uses the FREE figure.")
    return out


def budget(free_gib: float) -> dict:
    kvcfg = KVConfig()
    plan = DEFAULT.memory_plan(vram_gib=free_gib, usable_frac=0.97, bpw=1.75)
    per_tok = kvcfg.bytes_per_token(DEFAULT)
    plan["kv_bytes_per_token"] = per_tok
    plan["max_resident_context"] = int(max(plan["kv_free_bytes"], 0) // per_tok)
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out")
    ap.add_argument("--bandwidth", type=float,
                    help="measured GB/s, overrides the estimate")
    a = ap.parse_args()

    gpu, tc = gpu_info(), toolchain_info()
    print("=" * 72)
    print("  Bonsai 2 27B native runtime -- machine report")
    print("=" * 72)

    print("\nGPU")
    if not gpu.get("available"):
        print("  nvidia-smi not found or no GPU visible")
    else:
        for k in FIELDS:
            if gpu.get(k):
                print(f"  {k:<28} {gpu[k]}")
        if "est_bandwidth_gbs_128bit" in gpu:
            print(f"  {'est. bandwidth (128-bit bus)':<28} "
                  f"{gpu['est_bandwidth_gbs_128bit']} GB/s")

    print("\nToolchain")
    for k, v in tc.items():
        print(f"  {k:<28} {v}")

    print("\nVerdicts")
    for v in verdicts(gpu, tc) or ["  (nothing to report)"]:
        print(f"  - {v}")

    free = gpu.get("memory.free.gib") or tc.get("torch_free_gib") or 8.0
    bw = a.bandwidth or gpu.get("est_bandwidth_gbs_128bit") or 384.0
    p = budget(float(free))

    print(f"\nMemory budget (against {free:.2f} GiB free, PTQ1_0 weights, 4-bit KV)")
    for k in ("weight_bytes", "gdn_state_bytes", "overhead_bytes", "fixed_bytes",
              "kv_free_bytes"):
        print(f"  {k:<28} {p[k] / GIB:6.2f} GiB")
    print(f"  {'kv bytes/token':<28} {p['kv_bytes_per_token'] / 1024:6.1f} KiB")
    print(f"  {'MAX RESIDENT CONTEXT':<28} {p['max_resident_context']:>6,} tokens")
    if p["max_resident_context"] < 8192:
        print("  WARNING: under 8K resident. Close other GPU users, or plan on "
              "host-paged cold blocks.")

    print(f"\nProjected decode at {bw:.0f} GB/s (55% of roofline)")
    for ctx in (4096, 8192, 32768, 65536):
        if ctx <= max(p["max_resident_context"], 1):
            t = DEFAULT.decode_roofline(bw, ctx, 0.55, bpw=1.75,
                                        kv_bits=4) * (64 / 72)   # 4.75 vs 4.0 bpw
            print(f"  ctx {ctx:>6}  {t:5.1f} tok/s")
    print("\nTarget for a fluent agentic loop: >= 25 tok/s sustained at 32K.\n")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"gpu": gpu, "toolchain": tc, "budget": p, "bandwidth_gbs": bw},
            indent=2, default=str))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
