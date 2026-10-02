"""The refusal ablation, as engine policy rather than a module wrapper.

The MLX implementation wraps each residual writer in an ``Ablated`` module. That is the
right shape for a framework you do not control; it is the wrong shape for an engine you
do. Here the projection is a *property of the forward pass*, which buys three things
the wrapper cannot:

* **No per-site launch.** The correction fuses into the residual add the engine already
  performs, so 129 sites cost one extra block-wide reduction each and zero extra global
  memory traffic -- against two extra GEMVs per site for the LoRA path.
* **Runtime control.** ``alpha`` lives in a one-element device tensor, so it can be
  changed per request without reloading weights or recapturing a CUDA graph. A baked
  LoRA fixes its scale at load.
* **No module renaming.** ``install()`` in the MLX path renames ``down_proj`` to
  ``down_proj.inner``, which is why it must run *after* weight loading. Here there is
  nothing to rename and no ordering hazard.

Invariants carried over from the original implementation, each with a test:

* exactly **129** sites: 64 ``mlp.down_proj`` + 48 ``linear_attn.out_proj``
  + 16 ``self_attn.o_proj`` + ``model.embed_tokens``;
* ``alpha == 0`` is **bit-identical** to the base model;
* the direction is unit-norm and lives in the plain (unrotated) hidden basis, so no
  Hadamard handling is needed on this path;
* the reduction accumulates in fp32 regardless of activation dtype.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import DEFAULT, BonsaiConfig

F32 = np.float32


def load_direction(path: str | Path, hidden_size: int = 5120) -> np.ndarray:
    """Load and renormalise the refusal direction.

    Accepts the raw ``.bin`` (little-endian float32, no header) or a ``.safetensors``
    holding a tensor named ``direction``.
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.numpy import load_file
        except ImportError as e:                                  # pragma: no cover
            raise ImportError("pip install safetensors to read .safetensors") from e
        d = load_file(str(path))["direction"]
    else:
        d = np.fromfile(path, dtype="<f4")
    d = d.astype(F32).reshape(-1)
    if d.shape != (hidden_size,):
        raise ValueError(f"direction has {d.shape[0]} dims, expected {hidden_size}")
    n = np.linalg.norm(d)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("direction has degenerate norm")
    return d / n


@dataclass
class AblationPolicy:
    """Which sites are ablated, and how hard.

    ``alpha``  0 off · 0.5-0.9 partial · 1.0 matches a full weight orthogonalisation ·
    >1 over-projects. The direction was estimated on the bf16 base model, and transfer
    across quantisation-aware training is unmeasured, so sweeping alpha is expected.
    """
    direction: np.ndarray
    alpha: float = 1.0
    layers: frozenset | None = None        #: None = every layer
    include_embedding: bool = True
    cfg: BonsaiConfig = field(default=DEFAULT)

    def __post_init__(self):
        d = np.asarray(self.direction, dtype=F32).reshape(-1)
        if d.shape != (self.cfg.hidden_size,):
            raise ValueError(f"direction must be {self.cfg.hidden_size}-dim")
        norm = np.linalg.norm(d)
        if abs(norm - 1.0) > 1e-3:
            d = d / max(norm, 1e-12)
        self.direction = d
        if self.layers is not None and not isinstance(self.layers, frozenset):
            self.layers = frozenset(int(i) for i in self.layers)

    # ------------------------------------------------------------------- site logic
    def sites(self) -> list[str]:
        """The module paths this policy will ablate."""
        out = []
        for path in self.cfg.residual_writers():
            if path == self.cfg.embedding_path:
                if self.include_embedding:
                    out.append(path)
                continue
            if self.layers is not None:
                if int(path.split(".")[2]) not in self.layers:
                    continue
            out.append(path)
        return out

    def is_active(self, path: str) -> bool:
        return self.alpha != 0.0 and path in set(self.sites())

    # ------------------------------------------------------------------- the operator
    def apply(self, y: np.ndarray) -> np.ndarray:
        """``y - alpha * dot(y, r) * r`` with an fp32 interior.

        Exactly the operator the fused CUDA kernel implements; this is its oracle.
        """
        if self.alpha == 0.0:
            return y
        yf = y.astype(F32)
        comp = (yf * self.direction).sum(axis=-1, keepdims=True)
        return (yf - F32(self.alpha) * comp * self.direction).astype(y.dtype)

    def residual_add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The fused form: ``h = x + y - alpha*dot(y,r)*r``."""
        if self.alpha == 0.0:
            return (x.astype(F32) + y.astype(F32)).astype(y.dtype)
        return (x.astype(F32) + self.apply(y).astype(F32)).astype(y.dtype)

    # ------------------------------------------------------------------- diagnostics
    def residual_fraction(self, h: np.ndarray) -> float:
        """``|<h, r>| / ||h||`` -- what ``scripts/selfcheck.py`` measures.

        Measure *block outputs*, never the model's final output: the closing RMSNorm
        multiplies by a diagonal weight, which does not preserve orthogonality and
        reintroduces a component even after a perfect ablation.
        """
        v = h.astype(F32).reshape(-1)
        return float(abs(v @ self.direction) / max(np.linalg.norm(v), 1e-12))

    def describe(self) -> dict:
        s = self.sites()
        return {
            "alpha": self.alpha,
            "n_sites": len(s),
            "expected_sites": len(self.cfg.residual_writers()),
            "layers": "all" if self.layers is None else sorted(self.layers),
            "include_embedding": self.include_embedding,
            "direction_norm": float(np.linalg.norm(self.direction)),
            "breakdown": {
                "mlp.down_proj": sum(p.endswith("mlp.down_proj") for p in s),
                "linear_attn.out_proj": sum(p.endswith("linear_attn.out_proj") for p in s),
                "self_attn.o_proj": sum(p.endswith("self_attn.o_proj") for p in s),
                "embed_tokens": sum(p == self.cfg.embedding_path for p in s),
            },
        }


def from_repo(directions_dir: str | Path = "directions", alpha: float = 1.0,
              cfg: BonsaiConfig = DEFAULT, **kw) -> AblationPolicy:
    """Build a policy from this repository's shipped direction and metadata."""
    d = Path(directions_dir)
    meta_path = d / "direction.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    hidden = int(meta.get("hidden_size", cfg.hidden_size))
    direction = load_direction(d / "refusal_dir_fp32.bin", hidden)
    return AblationPolicy(direction=direction, alpha=alpha, cfg=cfg, **kw)
