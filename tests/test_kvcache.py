"""KV cache tests -- correctness, the accuracy cost of int4, and the budget that
decides whether the model is fluent on 8 GB."""
from __future__ import annotations

import numpy as np
import pytest

from bonsai import kvcache as kv
from bonsai.config import DEFAULT, GIB

F32 = np.float32


# ------------------------------------------------------------------- quantisation

@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("group", [32, 64, 128])
def test_quantize_round_trip_shapes(group, symmetric):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 4, 256)).astype(F32)
    c, s, z = kv.quantize(x, group, symmetric)
    assert c.shape == (5, 4, 256 // group, group // 2)
    assert s.shape == z.shape == (5, 4, 256 // group)
    out = kv.dequantize(c, s, z, group, symmetric)
    assert out.shape == x.shape


def test_nibble_packing_is_recoverable():
    """Two 4-bit codes per byte, low nibble first."""
    x = np.arange(16, dtype=F32).reshape(1, 16)
    c, s, z = kv.quantize(x, group=16, symmetric=False)
    assert c.shape == (1, 1, 8)
    out = kv.dequantize(c, s, z, 16, False)
    np.testing.assert_allclose(out, x, atol=0.6)     # 16 levels over a range of 15


def test_asymmetric_is_exact_on_16_levels():
    """A group whose values already sit on the 16-level grid must round-trip exactly."""
    x = (np.arange(64, dtype=F32) % 16).reshape(1, 64)
    out = kv.dequantize(*kv.quantize(x, 64, False), 64, False)
    np.testing.assert_allclose(out, x, atol=1e-3)


@pytest.mark.parametrize("group,max_rel", [(32, 0.085), (64, 0.095), (128, 0.105)])
def test_int4_error_matches_quantisation_theory(group, max_rel):
    """~10% elementwise error is inherent to 16 levels, not a defect.

    For a group of n roughly Gaussian values the min-max span is ~5.8 sigma, so the
    step is span/15 and the RMS error is step/sqrt(12) ~ 0.11 sigma. These bounds
    track that; a regression past them means the quantiser broke.
    """
    rng = np.random.default_rng(1)
    x = rng.standard_normal((64, 4, 256)).astype(F32)
    m = kv.measure_error(x, group, symmetric=False)
    assert m["rel_fro"] < max_rel, m
    assert m["cosine"] > 0.993, m


def test_finer_groups_are_monotonically_more_accurate():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((64, 4, 256)).astype(F32)
    e = [kv.measure_error(x, g, False)["rel_fro"] for g in (32, 64, 128)]
    assert e[0] < e[1] < e[2], e


def test_large_groups_degrade_badly_with_outliers():
    """Why group 128 is a trap: attention K/V have outlier channels, and a coarse
    group lets one spike stretch the scale for 128 values."""
    rng = np.random.default_rng(12)
    x = (rng.standard_normal((64, 4, 16)).astype(F32)
         @ rng.standard_normal((16, 256)).astype(F32))
    x[..., ::37] *= 6.0
    e32 = kv.measure_error(x, 32, False)["rel_fro"]
    e128 = kv.measure_error(x, 128, False)["rel_fro"]
    assert e128 > 1.5 * e32, (e32, e128)


def test_k_is_quantised_more_finely_than_v():
    """K errors feed the softmax and compound; V errors enter linearly."""
    c = kv.KVConfig()
    assert c.k_group < c.v_group
    assert kv.bits_per_value(c.k_group) > kv.bits_per_value(c.v_group)


def test_asymmetric_beats_symmetric_on_offset_data():
    """K tends to be poorly centred; that is why a zero point is worth 16 bits."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((32, 4, 256)).astype(F32) * 0.3 + 4.0
    a = kv.measure_error(x, 64, symmetric=False)
    s = kv.measure_error(x, 64, symmetric=True)
    assert a["rel_fro"] < s["rel_fro"]


def test_bits_per_value_matches_the_budget():
    assert kv.bits_per_value(64, False) == 4.5
    assert kv.bits_per_value(64, True) == 4.25
    assert kv.bits_per_value(128, False) == 4.25


# ------------------------------------------------------------------------- the cache

def test_layer_cache_round_trips_within_tolerance():
    rng = np.random.default_rng(3)
    c = kv.LayerKVCache(4, 256, kv.KVConfig(sink=8, page=16))
    K = rng.standard_normal((100, 4, 256)).astype(F32)
    V = rng.standard_normal((100, 4, 256)).astype(F32)
    c.append(K, V)
    assert c.length == 100
    gk, gv = c.gather()
    assert gk.shape == (4, 100, 256)
    np.testing.assert_allclose(gk, K.transpose(1, 0, 2), atol=0.25)
    np.testing.assert_allclose(gv, V.transpose(1, 0, 2), atol=0.25)


def test_sink_window_is_exact():
    """The most recent tokens must be bit-exact fp16, not quantised."""
    rng = np.random.default_rng(4)
    cfg = kv.KVConfig(sink=8, page=16)
    c = kv.LayerKVCache(2, 128, cfg)
    K = rng.standard_normal((64, 2, 128)).astype(F32)
    c.append(K, K.copy())
    gk, _ = c.gather()
    recent = K[-cfg.sink:].transpose(1, 0, 2).astype(np.float16).astype(F32)
    np.testing.assert_array_equal(gk[:, -cfg.sink:], recent)


def test_token_by_token_equals_bulk_append():
    rng = np.random.default_rng(5)
    cfg = kv.KVConfig(sink=4, page=8)
    K = rng.standard_normal((40, 2, 128)).astype(F32)
    V = rng.standard_normal((40, 2, 128)).astype(F32)
    bulk = kv.LayerKVCache(2, 128, cfg); bulk.append(K, V)
    step = kv.LayerKVCache(2, 128, cfg)
    for t in range(40):
        step.append(K[t:t + 1], V[t:t + 1])
    for a, b in zip(bulk.gather(), step.gather()):
        np.testing.assert_array_equal(a, b)


def test_cache_rejects_wrong_shape():
    c = kv.LayerKVCache(4, 256)
    with pytest.raises(ValueError):
        c.append(np.zeros((2, 3, 256), F32), np.zeros((2, 3, 256), F32))


def test_full_cache_covers_exactly_the_16_attention_layers():
    cache = kv.KVCache()
    assert len(cache.layers) == 16
    assert sorted(cache.layers) == [i for i in range(64) if i % 4 == 3]
    with pytest.raises(KeyError, match="linear-attention"):
        cache[0]


def test_cache_memory_is_within_a_few_percent_of_the_model():
    """Measured bytes must track the analytic budget, or the plan is fiction.

    Flush first: the analytic figure describes fully quantised storage, while a live
    cache also holds an unquantised sink window.
    """
    cache = kv.KVCache(kvcfg=kv.KVConfig(sink=0, page=128))
    rng = np.random.default_rng(6)
    n = 256
    for layer in cache.layers:
        cache[layer].append(
            rng.standard_normal((n, 4, 256)).astype(F32),
            rng.standard_normal((n, 4, 256)).astype(F32))
        cache[layer].flush()
    predicted = cache.kvcfg.bytes_per_token(DEFAULT) * n
    assert abs(cache.nbytes() - predicted) / predicted < 0.02


def test_sink_window_costs_what_it_should():
    """An unflushed cache holds the sink in fp16; the overhead must be bounded."""
    c = kv.LayerKVCache(4, 256, kv.KVConfig(sink=64, page=64))
    rng = np.random.default_rng(13)
    c.append(rng.standard_normal((512, 4, 256)).astype(F32),
             rng.standard_normal((512, 4, 256)).astype(F32))
    assert c.quantized_tokens() >= 512 - (64 + 64)
    before = c.nbytes()
    c.flush()
    assert c.nbytes() < before


# ---------------------------------------------------------------------- the budget

def test_int4_budget_beats_fp16_by_more_than_3x():
    fp16 = DEFAULT.kv_bytes_per_token(16)
    int4 = kv.KVConfig().bytes_per_token(DEFAULT)
    assert fp16 == 64 * 1024
    assert int4 == 19 * 1024          # K group 32 (5.0 bpw) + V group 64 (4.5 bpw)
    assert fp16 / int4 > 3.3


def test_resident_context_on_an_8gb_card():
    """The headline claim: 4-bit KV keeps a long context entirely in VRAM."""
    plan = kv.KVCache().plan(vram_gib=8.0)
    assert plan["max_resident_context"] > 60_000, plan
    assert plan["fixed_bytes"] / GIB < 6.5


def test_fp16_kv_does_not_fit_a_useful_context():
    """The situation being fixed: fp16 KV forces offload, which costs ~5x throughput."""
    plan = DEFAULT.memory_plan(vram_gib=8.0, kv_bits=16)
    assert plan["max_resident_context"] < 25_000


@pytest.mark.parametrize("ctx", [8192, 32768, 65536])
def test_decode_projection_is_usable_for_agentic_work(ctx):
    per_tok = (DEFAULT.weight_bytes(1.75) + 2 * DEFAULT.gdn_state_bytes(4)
               + ctx * kv.KVConfig().bytes_per_token(DEFAULT))
    tok_s = 384e9 * 0.55 / per_tok
    assert tok_s > 25, f"{ctx}: {tok_s:.1f} tok/s"
