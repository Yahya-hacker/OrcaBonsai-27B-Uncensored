"""Load the compiled CUDA kernels, or explain precisely why they are unavailable.

Two paths:

* **ahead of time** -- ``pip install -e .`` builds ``bonsai_kernels`` and we import it;
* **just in time** -- :func:`load` compiles the sources into a cache with
  ``torch.utils.cpp_extension.load``. First call takes a minute or two; later ones hit
  the cache.

The diagnostics matter more than they look. The dominant sm_120 failure mode is a
*runtime* ``no kernel image is available for execution on the device``, which says
nothing about the actual cause (a toolkit older than CUDA 12.8). :func:`check_toolchain`
turns that into a sentence you can act on.

See ``docs/BUILDING-CUDA.md``.
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
SOURCES = ["bindings.cpp", "ternary_gemv.cu", "residual_ablate.cu"]

#: Blackwell consumer (RTX 50-series, including the 5060 Laptop) is sm_120.
#: compute_120 PTX is emitted alongside so a future architecture can JIT forward.
DEFAULT_ARCH = "12.0"
NVCC_FLAGS = [
    "-O3", "--use_fast_math", "-std=c++17",
    "--expt-relaxed-constexpr", "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "-lineinfo",
]

_module = None


class ToolchainError(RuntimeError):
    pass


def check_toolchain(require_cuda: bool = True) -> dict:
    """Validate the stack *before* compiling, with actionable messages."""
    try:
        import torch
    except ImportError as e:                                        # pragma: no cover
        raise ToolchainError(
            "PyTorch is not installed. For Blackwell (RTX 50-series) you need a cu128 "
            "or newer build:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu128"
        ) from e

    info = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_home": os.environ.get("CUDA_HOME") or "/usr/local/cuda",
    }
    if not require_cuda:
        return info
    if not torch.cuda.is_available():                               # pragma: no cover
        raise ToolchainError(
            "torch.cuda.is_available() is False. Check `nvidia-smi` works and that the "
            "driver is R570 or newer (required for CUDA 12.8 / Blackwell)."
        )

    cap = torch.cuda.get_device_capability()
    info["device"] = torch.cuda.get_device_name()
    info["capability"] = cap

    tc = tuple(int(p) for p in (torch.version.cuda or "0.0").split(".")[:2])
    if cap[0] >= 12 and tc < (12, 8):                               # pragma: no cover
        raise ToolchainError(
            f"{info['device']} is sm_{cap[0]}{cap[1]} (Blackwell), but this PyTorch was "
            f"built against CUDA {torch.version.cuda}. sm_120 needs CUDA 12.8 or newer "
            f"-- 12.9 is recommended.\n"
            f"  pip install --force-reinstall torch "
            f"--index-url https://download.pytorch.org/whl/cu128\n"
            f"Symptom if you ignore this: 'no kernel image is available for execution "
            f"on the device' at runtime, not at build time."
        )
    return info


def arch_list() -> str:
    """The value for TORCH_CUDA_ARCH_LIST, defaulting to the detected device."""
    if "TORCH_CUDA_ARCH_LIST" in os.environ:
        return os.environ["TORCH_CUDA_ARCH_LIST"]
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"{major}.{minor}"
    except Exception:                                               # pragma: no cover
        pass
    return DEFAULT_ARCH


def load(verbose: bool = False, force: bool = False):
    """Return the kernel module, compiling on first use."""
    global _module
    if _module is not None and not force:
        return _module

    try:                                    # prefer an ahead-of-time build
        import bonsai_kernels                                       # type: ignore
        _module = bonsai_kernels
        return _module
    except ImportError:
        pass

    check_toolchain()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list())
    from torch.utils.cpp_extension import load as _load

    _module = _load(
        name="bonsai_kernels",
        sources=[str(_HERE / s) for s in SOURCES],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=NVCC_FLAGS,
        verbose=verbose,
    )
    return _module


def available() -> bool:
    try:
        load()
        return True
    except Exception:
        return False
