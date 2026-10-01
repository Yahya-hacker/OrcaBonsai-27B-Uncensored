// K7 -- fused residual add with refusal ablation.
//
// This is the kernel the whole project exists for, and the design went through one
// correction worth recording.
//
// The obvious idea is to fuse the projection into the matmul *epilogue*. That does not
// work. The operator is
//
//     y <- y - alpha * dot(y, r) * r
//
// and in a GEMV the 5120 output elements are spread across thread blocks, so dot(y, r)
// needs a grid-wide reduction. You cannot do that in an epilogue without a grid sync.
//
// The fix is better than the original plan: fuse into the **residual add**, which
// already has to read x, read y and write h. One block owns one token's entire hidden
// vector, so the reduction is block-local:
//
//     h = x + y - alpha * dot(y, r) * r
//
// Cost over a plain residual add: one block-wide reduction over 5120 floats. No extra
// global traffic at all. Across 129 sites that is genuinely free, where the LoRA path
// costs two extra GEMVs per site and the MLX path a separate compiled kernel per site.
//
// Numerics: the dot product and the correction accumulate in fp32 regardless of the
// activation dtype, matching bonsai_abliterate/ablation.py, which casts to fp32 before
// reducing. This is not optional -- see R4. At 129 sites a bf16 reduction would drift.
//
// alpha and r stay **runtime arguments**, never baked in. scripts/test_ablation.py
// depends on sweeping alpha without recompiling, and alpha == 0 must be bit-identical
// to the base model. That is also why CUDA-graph capture (R9) must capture a graph
// whose parameters are device pointers, not literals.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace bonsai {

constexpr int kWarp = 32;

__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x / kWarp;
    #pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    const int nwarps = blockDim.x / kWarp;
    v = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.0f;
    if (warp == 0) {
        #pragma unroll
        for (int off = kWarp / 2; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffffu, v, off);
        if (threadIdx.x == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

// One block per token. blockDim.x should divide hidden (1024 for hidden == 5120 gives
// 5 elements per thread, held in registers -- no shared memory for the vector itself).
//
//   x      [tokens, hidden]  residual stream in
//   y      [tokens, hidden]  writer output (down_proj / o_proj / out_proj)
//   r      [hidden]          unit-norm refusal direction, fp32
//   out    [tokens, hidden]  h = x + y - alpha*dot(y,r)*r
//
// Pass alpha_ptr (device) rather than a literal so a captured CUDA graph stays tunable.
template <typename T, int kMaxPerThread = 8>
__global__ void residual_add_ablate(
        const T* __restrict__ x, const T* __restrict__ y,
        const float* __restrict__ r, const float* __restrict__ alpha_ptr,
        T* __restrict__ out, int hidden) {
    extern __shared__ float smem[];
    const long base = (long)blockIdx.x * hidden;

    float yv[kMaxPerThread];
    float rv[kMaxPerThread];
    float partial = 0.0f;
    int n = 0;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x, ++n) {
        yv[n] = static_cast<float>(y[base + i]);
        rv[n] = r[i];
        partial = fmaf(yv[n], rv[n], partial);
    }

    const float dot = block_reduce_sum(partial, smem);
    const float k = (*alpha_ptr) * dot;

    n = 0;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x, ++n) {
        const float h = static_cast<float>(x[base + i]) + yv[n] - k * rv[n];
        out[base + i] = static_cast<T>(h);
    }
}

// Variant for the embedding, the one asymmetric site. A row of the embedding table *is*
// a residual vector, so there is no x to add: h = e - alpha*dot(e,r)*r.
template <typename T, int kMaxPerThread = 8>
__global__ void embed_ablate(
        const T* __restrict__ e, const float* __restrict__ r,
        const float* __restrict__ alpha_ptr, T* __restrict__ out, int hidden) {
    extern __shared__ float smem[];
    const long base = (long)blockIdx.x * hidden;

    float ev[kMaxPerThread], rv[kMaxPerThread];
    float partial = 0.0f;
    int n = 0;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x, ++n) {
        ev[n] = static_cast<float>(e[base + i]);
        rv[n] = r[i];
        partial = fmaf(ev[n], rv[n], partial);
    }
    const float k = (*alpha_ptr) * block_reduce_sum(partial, smem);
    n = 0;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x, ++n)
        out[base + i] = static_cast<T>(ev[n] - k * rv[n]);
}

template __global__ void residual_add_ablate<__half>(
    const __half*, const __half*, const float*, const float*, __half*, int);
template __global__ void residual_add_ablate<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*,
    __nv_bfloat16*, int);
template __global__ void embed_ablate<__half>(
    const __half*, const float*, const float*, __half*, int);
template __global__ void embed_ablate<__nv_bfloat16>(
    const __nv_bfloat16*, const float*, const float*, __nv_bfloat16*, int);

}  // namespace bonsai
