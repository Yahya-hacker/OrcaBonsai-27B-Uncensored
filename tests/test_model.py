"""End-to-end engine tests on a real (miniature) model.

Builds an actual ``.bonsai`` container with the full architecture at 1/5 scale, loads
it through the real loader, and runs the real forward pass and generation loop. Every
code path the 27B model uses is exercised -- packed projections, the Hadamard, both
mixer types, the GDN recurrence, the KV cache, sampling and the ablation -- just with
small tensors.

What this cannot test is the CUDA kernels, which need a GPU. It tests the portable
torch path, which is what those kernels must agree with.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bonsai import codec                                    # noqa: E402
from bonsai.config import BonsaiConfig                      # noqa: E402
from bonsai.format import BonsaiReader, BonsaiWriter, split_ptq1_0  # noqa: E402
from bonsai.generate import SamplingConfig, generate, sample        # noqa: E402
from bonsai.loader import ModelLoader, load_model           # noqa: E402
from bonsai.model import RuntimeConfig                      # noqa: E402
from bonsai.torch_ops import dequantize_ternary, fwht       # noqa: E402


def tiny_cfg() -> BonsaiConfig:
    """1/5-scale but structurally identical: 4 layers, layer 3 is full attention."""
    return BonsaiConfig(
        hidden_size=1024, num_layers=4, intermediate_size=2048, vocab_size=2048,
        max_position_embeddings=4096, num_attention_heads=4, num_key_value_heads=1,
        head_dim=256, linear_num_key_heads=4, linear_num_value_heads=8,
        linear_key_head_dim=128, linear_value_head_dim=128)


def _ternary(rng, out_f, in_f):
    codes = rng.integers(0, 3, size=(out_f, in_f), dtype=np.uint8)
    scales = (rng.random((out_f, in_f // 128)).astype(np.float16) * 0.05 + 0.01)
    raw = codec.encode_ptq1_0(codes, scales)
    return split_ptq1_0(raw, out_f, in_f)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    cfg = tiny_cfg()
    path = tmp_path_factory.mktemp("m") / "tiny.bonsai"
    rng = np.random.default_rng(0)

    with BonsaiWriter(path, {"arch": "qwen35", "quant": "PTQ1_0"}) as w:
        for name, out_f, in_f in cfg.packed_modules():
            c, s = _ternary(rng, out_f, in_f)
            w.add_ternary(name, c, s, (out_f, in_f))
        for name, size in cfg.unpacked_modules():
            if name.endswith(("input_layernorm", "post_attention_layernorm",
                              "model.norm", "linear_attn.norm")):
                a = np.ones(size, np.float32)
            elif name.endswith("A_log"):
                a = rng.standard_normal(size).astype(np.float32) * 0.1
            elif name.endswith("dt_bias"):
                a = np.zeros(size, np.float32)
            else:
                a = (rng.standard_normal(size).astype(np.float32) * 0.02)
            w.add_dense(name, a)

    rt = RuntimeConfig(device="cpu", dtype=torch.float32, alpha=0.0)
    model = ModelLoader(path, cfg, rt).build()
    return model, cfg, path


# ---------------------------------------------------------------- dequant parity

def test_torch_dequant_matches_the_numpy_oracle(tiny_model):
    """The single most important parity check: the torch path must reproduce the
    Tier-1 oracle exactly, or every downstream number is meaningless."""
    _, cfg, path = tiny_model
    r = BonsaiReader(path)
    for name, out_f, in_f in cfg.packed_modules()[:6]:
        codes, scales = r.ternary(name)
        got = dequantize_ternary(
            torch.from_numpy(np.ascontiguousarray(codes).reshape(out_f, -1)),
            torch.from_numpy(np.ascontiguousarray(scales)),
            out_f, in_f, torch.float32).numpy()
        np.testing.assert_array_equal(got, r.dequantize(name))


def test_dequant_levels_are_exactly_three(tiny_model):
    _, cfg, path = tiny_model
    r = BonsaiReader(path)
    name, out_f, in_f = cfg.packed_modules()[0]
    codes, scales = r.ternary(name)
    w = dequantize_ternary(
        torch.from_numpy(np.ascontiguousarray(codes).reshape(out_f, -1)),
        torch.from_numpy(np.ascontiguousarray(scales)), out_f, in_f, torch.float32)
    row = w[0] / scales[0, 0].astype(np.float32)
    assert set(np.unique(np.rint(row[:128].numpy()))) <= {-1.0, 0.0, 1.0}


def test_dequant_rejects_wrong_size(tiny_model):
    with pytest.raises(ValueError, match="expected"):
        dequantize_ternary(torch.zeros(10, dtype=torch.uint8),
                           torch.zeros(1, dtype=torch.float16), 4, 1024)


# ------------------------------------------------------------------------- fwht

def test_torch_fwht_matches_the_numpy_reference():
    from bonsai import reference as ref
    x = torch.randn(2, 2048)
    np.testing.assert_allclose(fwht(x, 1024).numpy(),
                               ref.fwht(x.numpy(), 1024), rtol=1e-4, atol=1e-4)


def test_torch_fwht_is_self_inverse():
    x = torch.randn(3, 1024)
    torch.testing.assert_close(fwht(fwht(x, 1024), 1024), x, rtol=1e-4, atol=1e-4)


# -------------------------------------------------------------------- the model

def test_model_builds_with_the_right_structure(tiny_model):
    model, cfg, _ = tiny_model
    assert len(model.blocks) == 4
    assert [b.is_full for b in model.blocks] == [False, False, False, True]
    from bonsai.model import BonsaiAttention, BonsaiGatedDelta
    assert isinstance(model.blocks[3].mixer, BonsaiAttention)
    assert isinstance(model.blocks[0].mixer, BonsaiGatedDelta)


def test_forward_produces_finite_logits(tiny_model):
    model, cfg, _ = tiny_model
    ids = torch.tensor([[1, 5, 9, 2, 7]])
    out = model(ids, model.new_caches(), model.new_states())
    assert out.shape == (1, 5, cfg.vocab_size)
    assert torch.isfinite(out).all()


def test_forward_is_deterministic(tiny_model):
    model, _, _ = tiny_model
    ids = torch.tensor([[3, 1, 4, 1, 5]])
    a = model(ids, model.new_caches(), model.new_states())
    b = model(ids, model.new_caches(), model.new_states())
    torch.testing.assert_close(a, b)


def test_incremental_decode_matches_full_prefill(tiny_model):
    """The cache and GDN state must make one-token-at-a-time identical to a single
    pass. This is the test that catches almost every state-management bug."""
    model, _, _ = tiny_model
    ids = [2, 8, 3, 11, 6, 4]

    full = model(torch.tensor([ids]), model.new_caches(), model.new_states())[0, -1]

    caches, states = model.new_caches(), model.new_states()
    model(torch.tensor([ids[:-1]]), caches, states, offset=0)
    step = model(torch.tensor([[ids[-1]]]), caches, states,
                 offset=len(ids) - 1)[0, -1]

    torch.testing.assert_close(full, step, rtol=2e-3, atol=2e-3)


def test_kv_cache_grows_only_on_attention_layers(tiny_model):
    model, cfg, _ = tiny_model
    caches, states = model.new_caches(), model.new_states()
    assert set(caches) == {3}
    assert set(states) == {0, 1, 2}
    model(torch.tensor([[1, 2, 3]]), caches, states)
    assert caches[3].length == 3


def test_gdn_state_is_context_independent_in_size(tiny_model):
    """48/64 layers carry a fixed-size state no matter how long the context is."""
    model, _, _ = tiny_model
    states = model.new_states()
    before = states[0].state.shape
    model(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]), model.new_caches(), states)
    assert states[0].state.shape == before
    assert torch.isfinite(states[0].state).all()


def test_gdn_state_actually_evolves(tiny_model):
    model, _, _ = tiny_model
    states = model.new_states()
    assert torch.count_nonzero(states[0].state) == 0
    model(torch.tensor([[4, 9, 2]]), model.new_caches(), states)
    assert torch.count_nonzero(states[0].state) > 0


# ---------------------------------------------------------------- the ablation

def test_alpha_zero_is_bit_identical_to_no_direction(tiny_model):
    """The acceptance criterion from selfcheck.py, at engine level."""
    model, cfg, _ = tiny_model
    ids = torch.tensor([[1, 2, 3, 4]])
    model.direction = None
    base = model(ids, model.new_caches(), model.new_states())

    d = np.random.default_rng(1).standard_normal(cfg.hidden_size).astype(np.float32)
    model.direction = torch.from_numpy(d / np.linalg.norm(d))
    model.set_alpha(0.0)
    off = model(ids, model.new_caches(), model.new_states())
    torch.testing.assert_close(base, off, rtol=0, atol=0)
    model.direction = None


def test_ablation_changes_the_output_and_scales_with_alpha(tiny_model):
    model, cfg, _ = tiny_model
    ids = torch.tensor([[1, 2, 3, 4]])
    d = np.random.default_rng(2).standard_normal(cfg.hidden_size).astype(np.float32)
    model.direction = torch.from_numpy(d / np.linalg.norm(d))

    model.set_alpha(0.0)
    a0 = model(ids, model.new_caches(), model.new_states())
    model.set_alpha(1.0)
    a1 = model(ids, model.new_caches(), model.new_states())
    model.set_alpha(2.0)
    a2 = model(ids, model.new_caches(), model.new_states())

    assert not torch.allclose(a0, a1)
    assert (a2 - a0).abs().mean() > (a1 - a0).abs().mean()
    model.direction = None
    model.set_alpha(0.0)


def test_alpha_is_tunable_without_reloading(tiny_model):
    """The whole point of runtime ablation: no weight edit, no reload."""
    model, cfg, _ = tiny_model
    d = np.random.default_rng(3).standard_normal(cfg.hidden_size).astype(np.float32)
    model.direction = torch.from_numpy(d / np.linalg.norm(d))
    ids = torch.tensor([[5, 6]])
    seen = set()
    for a in (0.0, 0.5, 1.0):
        model.set_alpha(a)
        seen.add(float(model(ids, model.new_caches(),
                             model.new_states()).sum().item()))
    assert len(seen) == 3
    model.direction = None
    model.set_alpha(0.0)


def test_site_count_is_two_per_block_plus_embedding(tiny_model):
    model, cfg, _ = tiny_model
    model.direction = torch.zeros(cfg.hidden_size)
    model.set_alpha(1.0)
    assert model.n_ablation_sites() == 2 * cfg.num_layers + 1
    model.direction = None
    model.set_alpha(0.0)


def test_scaled_to_27b_the_site_count_is_129():
    from bonsai.config import DEFAULT
    assert 2 * DEFAULT.num_layers + 1 == 129
    assert len(DEFAULT.residual_writers()) == 129


# ------------------------------------------------------------------- generation

def test_greedy_generation_is_reproducible(tiny_model):
    model, _, _ = tiny_model
    cfg = SamplingConfig(temperature=0.0, max_tokens=8)
    a = list(generate(model, [1, 2, 3], cfg, eos_ids=set()))
    b = list(generate(model, [1, 2, 3], cfg, eos_ids=set()))
    assert a == b and len(a) == 8


def test_sampling_respects_the_seed(tiny_model):
    model, _, _ = tiny_model
    c = SamplingConfig(temperature=0.8, top_p=0.9, max_tokens=6, seed=1234)
    assert list(generate(model, [4, 5], c, eos_ids=set())) == \
           list(generate(model, [4, 5], c, eos_ids=set()))


def test_generation_stops_at_eos(tiny_model):
    model, _, _ = tiny_model
    cfg = SamplingConfig(temperature=0.0, max_tokens=20)
    first = list(generate(model, [1, 2], cfg, eos_ids=set()))[0]
    assert list(generate(model, [1, 2], cfg, eos_ids={first})) == []


def test_sample_greedy_picks_the_argmax():
    logits = torch.tensor([0.1, 5.0, 0.3])
    assert sample(logits, SamplingConfig(temperature=0.0)) == 1


def test_top_k_restricts_the_support():
    logits = torch.tensor([10.0, 9.0, -50.0, -60.0])
    c = SamplingConfig(temperature=1.0, top_k=2, top_p=1.0, seed=0)
    g = torch.Generator().manual_seed(0)
    assert {sample(logits.clone(), c, gen=g) for _ in range(40)} <= {0, 1}


def test_repetition_penalty_discourages_repeats():
    logits = torch.tensor([5.0, 4.9, 0.0])
    c = SamplingConfig(temperature=0.0, repetition_penalty=2.0)
    assert sample(logits.clone(), c, generated=[0]) == 1


# ------------------------------------------------------------------ public API

def test_load_model_installs_the_shipped_direction(tiny_model):
    """load_model is the one entry point the CLI uses."""
    _, cfg, path = tiny_model
    d = np.random.default_rng(4).standard_normal(cfg.hidden_size).astype("<f4")
    d /= np.linalg.norm(d)
    p = path.parent / "dir.bin"
    d.tofile(p)
    m = load_model(path, direction=p, alpha=0.7, device="cpu",
                   dtype=torch.float32, cfg=cfg)
    assert m.alpha == 0.7
    assert m.direction is not None
    torch.testing.assert_close(m.direction, torch.from_numpy(d), rtol=1e-6, atol=1e-6)
    assert torch.isfinite(m(torch.tensor([[1, 2]]), m.new_caches(),
                            m.new_states())).all()
