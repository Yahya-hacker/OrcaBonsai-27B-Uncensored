// K9 -- 4-bit KV cache quantisation. Promoted to the critical path by the memory
// analysis: on 8 GB this is what keeps the cache in VRAM instead of across PCIe.
//
// STATUS: reference implementation, never compiled. Arithmetic mirrors
// bonsai/kvcache.py, which is tested.
//
// Two kernels, because the two directions have completely different shapes:
//
//   quantize_append   one new token per decode step. Tiny, latency-bound, trivially
//                     parallel: one warp per group.
//   dequantize_block  reads a page back for attention. Bandwidth-bound.
//
// Group sizes differ for K and V on purpose. K errors perturb attention logits, which
// are then exponentiated, so they compound; V errors enter the output linearly and
// partly average out. K therefore uses group 32 (5.00 bpw) and V group 64 (4.50 bpw).
// Measured: at group 128 the error on outlier-heavy data is ~80% worse than group 32,
// for a saving of 0.75 bits. Not worth it.
//
// Note on reading the cache in attention
// --------------------------------------
// Stock SDPA and flash-attn cannot read a 4-bit cache. The first version dequantises a
// page into an fp16 scratch buffer and calls SDPA normally -- that costs one extra
// read+write of the page, which is still roughly an order of magnitude cheaper than
// the PCIe transfer it replaces. Fusing dequantisation into an attention inner loop is
// an M5 optimisation, worth doing only if profiling says the scratch pass matters.

#include <cuda_fp16.h>

namespace bonsai {

constexpr int kKvWarp = 32;

// Quantise one token's worth of [heads, dim] into affine int4, one warp per group.
//
//   x       [heads, dim]       fp16/fp32 input, dim % group == 0
//   codes   [heads, dim/group, group/2]  two nibbles per byte, low nibble first
//   scale   [heads, dim/group] fp16
//   zero    [heads, dim/group] fp16
//
// grid (dim/group, heads), block (32).
__global__ void kv_quantize(const __half* __restrict__ x, uint8_t* __restrict__ codes,
                            __half* __restrict__ scale, __half* __restrict__ zero,
                            int dim, int group, bool symmetric) {
    const int head = blockIdx.y;
    const int grp  = blockIdx.x;
    const int lane = threadIdx.x;
    const int n_groups = dim / group;
    const __half* src = x + (size_t)head * dim + (size_t)grp * group;

    // ---- min / max across the group
    float lo = 3.4e38f, hi = -3.4e38f, amax = 0.0f;
    for (int i = lane; i < group; i += kKvWarp) {
        const float v = __half2float(src[i]);
        lo = fminf(lo, v); hi = fmaxf(hi, v); amax = fmaxf(amax, fabsf(v));
    }
    #pragma unroll
    for (int off = kKvWarp / 2; off > 0; off >>= 1) {
        lo   = fminf(lo,   __shfl_xor_sync(0xffffffffu, lo,   off));
        hi   = fmaxf(hi,   __shfl_xor_sync(0xffffffffu, hi,   off));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }

    float s, z;
    if (symmetric) { s = fmaxf(amax, 1e-8f) / 7.0f;      z = 0.0f; }
    else           { s = fmaxf(hi - lo, 1e-8f) / 15.0f;  z = lo;   }
    const float inv = 1.0f / s;

    const size_t gidx = (size_t)head * n_groups + grp;
    if (lane == 0) { scale[gidx] = __float2half(s); zero[gidx] = __float2half(z); }

    // ---- pack two codes per byte; lane i owns byte i, i.e. elements 2i and 2i+1
    uint8_t* dst = codes + gidx * (group / 2);
    for (int i = lane; i < group / 2; i += kKvWarp) {
        float a = __half2float(src[2 * i]);
        float b = __half2float(src[2 * i + 1]);
        int qa, qb;
        if (symmetric) {
            qa = __float2int_rn(a * inv) + 8;
            qb = __float2int_rn(b * inv) + 8;
        } else {
            qa = __float2int_rn((a - z) * inv);
            qb = __float2int_rn((b - z) * inv);
        }
        qa = min(max(qa, 0), 15);
        qb = min(max(qb, 0), 15);
        dst[i] = (uint8_t)(qa | (qb << 4));
    }
}

// Expand a page back to fp16 for SDPA. grid (dim/group, heads*tokens), block (32).
__global__ void kv_dequantize(const uint8_t* __restrict__ codes,
                              const __half* __restrict__ scale,
                              const __half* __restrict__ zero,
                              __half* __restrict__ out,
                              int dim, int group, bool symmetric) {
    const int row  = blockIdx.y;              // flattened (token, head)
    const int grp  = blockIdx.x;
    const int lane = threadIdx.x;
    const int n_groups = dim / group;
    const size_t gidx = (size_t)row * n_groups + grp;

    const float s = __half2float(scale[gidx]);
    const float z = __half2float(zero[gidx]);
    const uint8_t* src = codes + gidx * (group / 2);
    __half* dst = out + (size_t)row * dim + (size_t)grp * group;

    for (int i = lane; i < group / 2; i += kKvWarp) {
        const uint8_t byte = src[i];
        const float a = (float)(byte & 0x0F);
        const float b = (float)(byte >> 4);
        if (symmetric) {
            dst[2 * i]     = __float2half((a - 8.0f) * s);
            dst[2 * i + 1] = __float2half((b - 8.0f) * s);
        } else {
            dst[2 * i]     = __float2half(a * s + z);
            dst[2 * i + 1] = __float2half(b * s + z);
        }
    }
}

void kv_quantize_launch(const void* x, uint8_t* codes, void* scale, void* zero,
                        int heads, int dim, int group, bool symmetric,
                        cudaStream_t stream) {
    dim3 grid(dim / group, heads);
    kv_quantize<<<grid, kKvWarp, 0, stream>>>(
        (const __half*)x, codes, (__half*)scale, (__half*)zero, dim, group, symmetric);
}

void kv_dequantize_launch(const uint8_t* codes, const void* scale, const void* zero,
                          void* out, int rows, int dim, int group, bool symmetric,
                          cudaStream_t stream) {
    dim3 grid(dim / group, rows);
    kv_dequantize<<<grid, kKvWarp, 0, stream>>>(
        codes, (const __half*)scale, (const __half*)zero, (__half*)out,
        dim, group, symmetric);
}

}  // namespace bonsai
