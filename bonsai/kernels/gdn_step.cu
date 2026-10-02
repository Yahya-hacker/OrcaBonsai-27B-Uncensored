// K4 -- gated delta net decode step. The kernel MLX does not have on CUDA.
//
// STATUS: reference implementation, never compiled. Arithmetic mirrors
// bonsai/reference.py::gdn_step, which is tested.
//
// Why this kernel exists
// ----------------------
// mlx-lm's gated-delta kernel is Metal-only -- it literally returns None unless
// mx.metal.is_available() -- so on CUDA it falls back to a Python loop over
// timesteps. For 48 of 64 layers that is fatal for prefill and merely bad for decode.
// Writing it is a large part of what "native" buys.
//
// The recurrence, per value head h (48 of them), fp32 throughout (R4):
//
//     state[h] *= g[h]                        // [dv, dk], decay
//     kv[h]     = state[h] . k[h]             // [dv]
//     delta[h]  = (v[h] - kv[h]) * beta[h]    // [dv]
//     state[h] += outer(delta[h], k[h])
//     y[h]      = state[h] . q[h]             // [dv]
//
// q and k have 16 heads, v and the state have 48, so each key head serves 3 value
// heads (nv/nk = 3). That asymmetry is also why GGUF imports need a value-head
// permutation -- see bonsai/config.py::value_head_permutation (R1).
//
// Launch geometry follows the Metal kernel that is known to work on this model:
// grid (32, Dv, B*Hv), threadgroup (32, 4, 1) -- i.e. 32 lanes spanning Dk, four dv
// rows per block, one block group per (batch, value head). Each (h, dv) row needs a
// reduction over dk, which is exactly one warp-width shuffle reduction.
//
// State traffic is the thing to watch: [48, 128, 128] fp32 is 3 MiB per layer and
// 144 MiB total, read and written every token. At 384 GB/s that is ~0.8 ms/token
// across all 48 layers -- small beside the 5.53 GiB of weights, but it is pure
// overhead if the state is ever allowed to spill.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace bonsai {

constexpr int kGdnWarp = 32;

// One block handles kRows value-dim rows of one value head.
//
//   state  [nv, dv, dk]  fp32, updated in place
//   q, k   [nk, dk]      fp32 (already rms-normed and scaled: inv^2 on q, inv on k)
//   v      [nv, dv]      fp32
//   g      [nv]          decay in (0, 1]
//   beta   [nv]          gate
//   y      [nv, dv]      output
//
// grid (dv / kRows, nv), block (32, kRows).
template <int kRows = 4>
__global__ void gdn_decode_step(
        float* __restrict__ state, const float* __restrict__ q,
        const float* __restrict__ k, const float* __restrict__ v,
        const float* __restrict__ g, const float* __restrict__ beta,
        float* __restrict__ y, int nv, int nk, int dv, int dk) {
    const int head = blockIdx.y;                 // value head, 0..nv-1
    const int row  = blockIdx.x * kRows + threadIdx.y;   // dv index
    if (head >= nv || row >= dv) return;

    const int khead = head / (nv / nk);          // 3 value heads share a key head
    const int lane  = threadIdx.x;

    const float decay = g[head];
    const float b     = beta[head];
    float* srow = state + ((size_t)head * dv + row) * dk;

    // ---- decay, then kv = state . k, with 32 lanes striding dk
    float partial = 0.0f;
    for (int d = lane; d < dk; d += kGdnWarp) {
        const float s = srow[d] * decay;
        srow[d] = s;                              // write back the decayed state
        partial = fmaf(s, k[khead * dk + d], partial);
    }
    #pragma unroll
    for (int off = kGdnWarp / 2; off > 0; off >>= 1)
        partial += __shfl_xor_sync(0xffffffffu, partial, off);
    const float kv = partial;                     // broadcast via xor-reduction

    // ---- delta = (v - kv) * beta ; state += outer(delta, k) ; y = state . q
    const float delta = (v[head * dv + row] - kv) * b;
    float acc = 0.0f;
    for (int d = lane; d < dk; d += kGdnWarp) {
        const float s = fmaf(delta, k[khead * dk + d], srow[d]);
        srow[d] = s;
        acc = fmaf(s, q[khead * dk + d], acc);
    }
    #pragma unroll
    for (int off = kGdnWarp / 2; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, off);

    if (lane == 0) y[head * dv + row] = acc;
}

// g = exp(-exp(A_log) * softplus(a + dt_bias)), fp32 (R4). One thread per head.
__global__ void gdn_decay(const float* __restrict__ a, const float* __restrict__ A_log,
                          const float* __restrict__ dt_bias, float* __restrict__ g,
                          int nv) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nv) return;
    const float x = a[i] + dt_bias[i];
    // softplus, numerically stable for large |x|
    const float sp = (x > 20.0f) ? x : log1pf(__expf(x));
    g[i] = __expf(-__expf(A_log[i]) * sp);
}

template __global__ void gdn_decode_step<4>(
    float*, const float*, const float*, const float*, const float*, const float*,
    float*, int, int, int, int);

void gdn_step_launch(float* state, const float* q, const float* k, const float* v,
                     const float* g, const float* beta, float* y,
                     int nv, int nk, int dv, int dk, cudaStream_t stream) {
    constexpr int kRows = 4;
    dim3 grid((dv + kRows - 1) / kRows, nv);
    dim3 block(kGdnWarp, kRows);
    gdn_decode_step<kRows><<<grid, block, 0, stream>>>(
        state, q, k, v, g, beta, y, nv, nk, dv, dk);
}

void gdn_decay_launch(const float* a, const float* A_log, const float* dt_bias,
                      float* g, int nv, cudaStream_t stream) {
    gdn_decay<<<(nv + 127) / 128, 128, 0, stream>>>(a, A_log, dt_bias, g, nv);
}

}  // namespace bonsai
