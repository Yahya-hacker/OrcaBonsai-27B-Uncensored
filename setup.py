"""Ahead-of-time build for the native Bonsai CUDA kernels.

    pip install -e .                     # auto-detects your GPU's architecture
    TORCH_CUDA_ARCH_LIST=12.0 pip install -e .   # or pin it

Skips the CUDA extension entirely when no toolkit is present, so the pure-Python parts
(codec, container, converter, reference model, all 56 tests) still install and run on a
machine without a GPU. See docs/BUILDING-CUDA.md.
"""
from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "bonsai" / "kernels"

ext_modules, cmdclass = [], {}
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    if torch.cuda.is_available() or "TORCH_CUDA_ARCH_LIST" in __import__("os").environ:
        from bonsai.kernels import NVCC_FLAGS, SOURCES, arch_list
        import os

        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list())
        print(f"[bonsai] building CUDA kernels for sm_{os.environ['TORCH_CUDA_ARCH_LIST']}")
        ext_modules = [CUDAExtension(
            name="bonsai_kernels",
            sources=[str(KERNELS / s) for s in SOURCES],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": NVCC_FLAGS},
        )]
        cmdclass = {"build_ext": BuildExtension}
    else:
        print("[bonsai] no CUDA device or TORCH_CUDA_ARCH_LIST; skipping kernels")
except ImportError:
    print("[bonsai] torch not installed; installing pure-Python components only")

setup(
    name="bonsai-native",
    version="0.1.0",
    description="Native CUDA runtime for Ternary Bonsai 2 27B with runtime refusal ablation",
    packages=find_packages(include=["bonsai", "bonsai.*", "tools"]),
    package_data={"bonsai.kernels": ["*.cu", "*.cpp"]},
    python_requires=">=3.10",
    install_requires=["numpy>=1.26"],
    extras_require={"cuda": ["torch>=2.7"], "dev": ["pytest>=8.0"]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
