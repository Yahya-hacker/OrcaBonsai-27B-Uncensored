"""Native CUDA runtime for Ternary Bonsai 2 27B with runtime refusal ablation.

Separate from ``bonsai_abliterate`` (the original MLX/Metal path), which is left
untouched and serves as a correctness oracle until parity is reached.
"""
from . import codec, config, format  # noqa: F401

__all__ = ["codec", "config", "format"]
